from copy import deepcopy
from pathlib import Path
import unittest

import yaml

from mp299_ops.deployment import validate_manifests


ROOT = Path(__file__).resolve().parents[1]


class DeploymentContractTests(unittest.TestCase):
    def setUp(self):
        self.deployment = yaml.safe_load((ROOT / "k8s/deployment.yaml").read_text())
        self.service = yaml.safe_load((ROOT / "k8s/service.yaml").read_text())

    def test_repository_manifests_meet_client_contract(self):
        self.assertEqual(validate_manifests(self.deployment, self.service), [])

    def test_service_cannot_select_the_wrong_pods(self):
        service = deepcopy(self.service)
        service["spec"]["selector"]["app"] = "another-project"
        self.assertIn("Service selector must match the Pod labels",
                      validate_manifests(self.deployment, service))

    def test_wrong_image_and_replica_count_are_rejected(self):
        deployment = deepcopy(self.deployment)
        deployment["spec"]["replicas"] = 1
        deployment["spec"]["template"]["spec"]["containers"][0]["image"] = "mp299-api:latest"
        errors = validate_manifests(deployment, self.service)
        self.assertIn("Deployment must have exactly four replicas", errors)
        self.assertIn("API image must be mp299-api:1.0", errors)

    def test_service_cannot_forward_to_the_wrong_port(self):
        service = deepcopy(self.service)
        service["spec"]["ports"][0]["targetPort"] = 8080
        self.assertIn("Service must forward port 80 to port 8000",
                      validate_manifests(self.deployment, service))

    def test_both_health_probes_are_required(self):
        for name in ["readinessProbe", "livenessProbe"]:
            with self.subTest(probe=name):
                deployment = deepcopy(self.deployment)
                del deployment["spec"]["template"]["spec"]["containers"][0][name]
                self.assertIn(f"{name} must use /health on port 8000",
                              validate_manifests(deployment, self.service))

    def test_liveness_cannot_start_before_readiness(self):
        deployment = deepcopy(self.deployment)
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        container["livenessProbe"]["initialDelaySeconds"] = 5
        self.assertIn("Liveness must wait longer than readiness",
                      validate_manifests(deployment, self.service))

    def test_ci_runs_pytest_and_installs_requirements_on_push(self):
        workflow = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
        self.assertIn("push", workflow["on"])
        self.assertEqual(workflow["permissions"]["contents"], "read")
        steps = workflow["jobs"]["tests"]["steps"]
        commands = [step.get("run", "") for step in steps]
        self.assertTrue(any("-r requirements.txt" in command for command in commands))
        self.assertTrue(any("pytest" in command for command in commands))


if __name__ == "__main__":
    unittest.main()
