"""Check the Kubernetes files against PROBLEM.md sections 5 and 6."""
from pathlib import Path
import sys

import yaml


def validate_manifests(deployment, service):
    errors = []
    if deployment.get("apiVersion") != "apps/v1" or deployment.get("kind") != "Deployment":
        errors.append("Expected an apps/v1 Deployment")
    if service.get("apiVersion") != "v1" or service.get("kind") != "Service":
        errors.append("Expected a v1 Service")
    if deployment.get("metadata", {}).get("name") != "mp299-api":
        errors.append("Deployment name must be mp299-api")
    if service.get("metadata", {}).get("name") != "mp299-svc":
        errors.append("Service name must be mp299-svc")

    spec = deployment.get("spec", {})
    if type(spec.get("replicas")) is not int or spec["replicas"] != 4:
        errors.append("Deployment must have exactly four replicas")
    template = spec.get("template", {})
    labels = template.get("metadata", {}).get("labels", {})
    selector = spec.get("selector", {}).get("matchLabels", {})
    service_spec = service.get("spec", {})
    for name, selected in [("Deployment", selector), ("Service", service_spec.get("selector", {}))]:
        if not selected or any(labels.get(key) != value for key, value in selected.items()):
            errors.append(f"{name} selector must match the Pod labels")

    containers = template.get("spec", {}).get("containers", [])
    api = next((item for item in containers if item.get("name") == "api"), None)
    if api is None:
        errors.append("Deployment must include the api container")
    else:
        if api.get("image") != "mp299-api:1.0":
            errors.append("API image must be mp299-api:1.0")
        if not any(port.get("containerPort") == 8000 for port in api.get("ports", [])):
            errors.append("API container must expose port 8000")
        for probe_name in ["readinessProbe", "livenessProbe"]:
            probe = api.get(probe_name, {})
            request = probe.get("httpGet", {})
            if request.get("path") != "/health" or request.get("port") != 8000:
                errors.append(f"{probe_name} must use /health on port 8000")
        ready_delay = api.get("readinessProbe", {}).get("initialDelaySeconds", 0)
        live_delay = api.get("livenessProbe", {}).get("initialDelaySeconds", 0)
        if live_delay <= ready_delay:
            errors.append("Liveness must wait longer than readiness")

    if not any(port.get("port") == 80 and port.get("targetPort") == 8000
               for port in service_spec.get("ports", [])):
        errors.append("Service must forward port 80 to port 8000")
    return errors


def main():
    root = Path(__file__).resolve().parents[2]
    with (root / "k8s/deployment.yaml").open(encoding="utf-8") as stream:
        deployment = yaml.safe_load(stream)
    with (root / "k8s/service.yaml").open(encoding="utf-8") as stream:
        service = yaml.safe_load(stream)
    errors = validate_manifests(deployment, service)
    if errors:
        print("\n".join(errors))
        return 1
    print("MP299 deployment contract passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
