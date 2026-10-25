# Phase 2 — Kubernetes Deployment and CI/CD Reliability

## Overview

This phase deploys the ML Prediction Service to a local Kubernetes cluster, configures application health checks and resource limits, validates API requests, and demonstrates scaling and recovery.

The project also uses GitHub Actions to automate linting, automated tests, and Docker image builds. Model releases are tracked through version metadata and evaluation metrics.

No production infrastructure or cloud deployment is required.

## Technology Stack

- Python 3.12
- Flask
- scikit-learn
- Docker
- Kubernetes
- Minikube
- kubectl
- GitHub Actions
- uv
- pytest
- Ruff

## Project Structure

```text
Task-2-MLOps-ML-Prediction-Service/
├── .github/
│   └── workflows/
│       └── ci.yml
├── Phase-1-ML-Service/
│   ├── app.py
│   ├── train.py
│   ├── tests/
│   │   └── test_app.py
│   ├── model/
│   │   ├── metrics.json
│   │   └── text_classifier.joblib
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── uv.lock
└── Phase-2-Kubernetes/
    ├── namespace.yaml
    ├── configmap.yaml
    ├── depolyment.yaml
    ├── service.yaml
    ├── build-v1.1/
    │   └── model/
    └── README.md
```

Note: `depolyment.yaml` is the existing filename and is intentionally referenced as it currently exists in the project.

## 1. Kubernetes Deployment

### Local cluster and namespace

Minikube was used to run a local Kubernetes cluster. The application is isolated in the `ml-prediction` namespace.

Check the cluster and namespace:

```powershell
minikube status
kubectl get nodes
kubectl get namespaces
kubectl get pods -n ml-prediction
```

Create the namespace if it does not already exist:

```powershell
kubectl apply -f .\Phase-2-Kubernetes\namespace.yaml
```

### Deployment and Service

The Kubernetes Deployment manages the prediction API replicas. The Service provides a stable endpoint for accessing the pods.

Apply the manifests:

```powershell
kubectl apply -f .\Phase-2-Kubernetes\namespace.yaml
kubectl apply -f .\Phase-2-Kubernetes\configmap.yaml
kubectl apply -f .\Phase-2-Kubernetes\depolyment.yaml
kubectl apply -f .\Phase-2-Kubernetes\service.yaml
```

Check the deployment and service:

```powershell
kubectl get deployments -n ml-prediction
kubectl get pods -n ml-prediction -o wide
kubectl get services -n ml-prediction
kubectl get endpoints -n ml-prediction
```

### Image version and ConfigMap

The Kubernetes Deployment uses the locally available Docker image `task2-ml-service:1.1`.

The ConfigMap `ml-prediction-config` contains non-secret settings, including the application environment and model metrics path. It does not contain credentials or API keys.

The application obtains its reported model version from the model metrics metadata. The `MODEL_VERSION` ConfigMap entry should not be treated as the source of truth for the model version unless the application is explicitly changed to read it.

### Health probes

Readiness and liveness probes use the API's `/health` endpoint on port `5001`.

- **Readiness probe:** determines whether the pod is ready to receive traffic.
- **Liveness probe:** checks whether the application is responding and may trigger a container restart when it repeatedly fails.

A broken readiness path was tested. The failed probe returned HTTP 404 and caused the pod to report `Ready=False`. Restoring the correct probe configuration returned the pods to a healthy state.

### CPU and memory requests and limits

The Deployment uses these resource settings:

| Resource | Request | Limit |
|---|---:|---:|
| CPU | 100m | 500m |
| Memory | 256Mi | 512Mi |

Requests reserve a reasonable starting allocation for each replica. Limits help prevent a container from consuming excessive node resources.

These are development settings for a local cluster, not production capacity guarantees. Actual requirements should be adjusted after measuring workload usage.

### Expose the service locally

Start the local service tunnel:

```powershell
minikube service ml-prediction-service -n ml-prediction
```

Keep the tunnel terminal open while testing. Use the URL printed by Minikube.

### API verification

Check the health endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:<PORT>/health
```

Replace `<PORT>` with the port in the URL printed by Minikube.

Expected health information includes a successful status, `model_loaded: true`, and model version `1.1.0`.

Test a valid prediction:

```powershell
$body = @{
    text = "NASA scientists discovered new planets and space technology"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri http://127.0.0.1:<PORT>/predict `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

The response should include a predicted class and model version.

Test invalid input:

```powershell
$body = @{ text = "" } | ConvertTo-Json

try {
    Invoke-RestMethod `
        -Uri http://127.0.0.1:<PORT>/predict `
        -Method Post `
        -ContentType "application/json" `
        -Body $body
}
catch {
    Write-Output $_.Exception.Message
}
```

An empty text input was verified to return HTTP 400. This demonstrates input validation for that invalid request.

## 2. CI/CD and Reliability

### GitHub Actions workflow

The workflow is stored at `.github/workflows/ci.yml` in the repository root.

On pushes and pull requests targeting `main`, it:

1. Checks out the repository.
2. Sets up Python 3.12 and uv.
3. Installs locked project dependencies.
4. Runs Ruff lint checks.
5. Runs automated pytest tests.
6. Builds the Docker image without pushing it to a registry.

The workflow uses no deployment credentials or secrets and does not deploy to a remote environment. Kubernetes deployment remains a local operation.

The latest reported GitHub Actions run passed. The Phase 1 test suite also passed all 24 tests locally when run with the project's virtual-environment Python interpreter.

### Release version tracking

Model releases are identified by version metadata and their associated evaluation metrics.

| Metric | v1.0.0 | v1.1.0 |
|---|---:|---:|
| Test accuracy | 88.07% | 87.94% |
| Training samples | 2,983 | 2,983 |
| Test samples | 746 | 746 |
| TF-IDF maximum features | 20,000 | 30,000 |

Version `1.1.0` was created by retraining with a larger TF-IDF feature limit. Its measured accuracy was slightly lower than v1.0.0, so the new version should not be described as a performance improvement.

The v1.1.0 model artifacts and metrics were copied into the Kubernetes build context, built into a Docker image, and deployed using the image tag `task2-ml-service:1.1`.

The API health endpoint and prediction response expose the model version, while the metrics file records the evaluation results.

### Rolling update and rollback

The Kubernetes Deployment was updated from the v1.0 image (`task2-ml-service:1.0`) to the v1.1 image (`task2-ml-service:1.1`). The rollout completed successfully, and both replicas became ready.

A rollback was then performed to the earlier working v1.0 image using the Kubernetes Deployment revision history. The v1.0 deployment reached 2/2 available replicas. After verifying the rollback, the v1.1 image was redeployed. The final Deployment returned to 2/2 available replicas running `task2-ml-service:1.1`.

This exercise demonstrates how Kubernetes supports reverting to an earlier working release and redeploying a newer version. After each release change, the deployed image, API-reported model version, and evaluation metrics should be checked.

Useful commands for checking deployment releases and rollout history:

```powershell
kubectl rollout status deployment/ml-prediction-api -n ml-prediction
kubectl rollout history deployment/ml-prediction-api -n ml-prediction
kubectl get deployment ml-prediction-api -n ml-prediction -o wide
```

To roll back to the previous Deployment revision:

```powershell
kubectl rollout undo deployment/ml-prediction-api -n ml-prediction
kubectl rollout status deployment/ml-prediction-api -n ml-prediction
kubectl get deployment ml-prediction-api -n ml-prediction -o wide
```

The previous revision should be verified before rolling back to ensure it is the intended working release. A successful rollout confirms deployment availability, but it does not guarantee model quality. The reported model version and evaluation metrics must also be reviewed.

### Scaling and service availability

The prediction API is stateless, so multiple replicas can serve requests without maintaining session state between pods.

The Deployment was scaled to two replicas. Both pods reached `1/1 Running`, and the Deployment reported `2/2` available replicas with service endpoints for both pods.

Check the current state:

```powershell
kubectl get deployment ml-prediction-api -n ml-prediction
kubectl get pods -n ml-prediction -o wide
kubectl get endpoints ml-prediction-service -n ml-prediction
```

If a pod is deleted, the Deployment controller should create a replacement to maintain the desired replica count.

### Pod recovery and troubleshooting

A running pod was deleted, and Kubernetes recreated a replacement pod.

Useful diagnostic commands:

```powershell
kubectl get pods -n ml-prediction -o wide
kubectl describe pods -n ml-prediction
kubectl get events -n ml-prediction --sort-by=.lastTimestamp
kubectl logs deployment/ml-prediction-api -n ml-prediction
```

- **Status:** identifies pod readiness, restart counts, and whether replacement pods are running.
- **Events:** explain scheduling, image, probe, and container lifecycle issues.
- **Logs:** help identify application startup errors and request-processing problems.

The replacement pod demonstrates controller-based recovery. Collect the corresponding status, events, and logs when preparing final evidence.

### Bad image or readiness-probe troubleshooting

A broken readiness probe was simulated by changing the health-check path to an invalid endpoint. Kubernetes reported an HTTP 404 probe failure and the pod became not ready.

The Deployment manifest was restored and reapplied, after which the pods returned to a healthy state.

A separate bad-image simulation has not been confirmed as completed. Do not report it as completed unless it is actually performed and verified.

## 3. Reproducibility

To reproduce the project on a compatible local development machine:

1. Install Python 3.12, uv, Docker Desktop, Minikube, and kubectl.
2. Clone the repository and enter its root directory.
3. Install the locked Phase 1 dependencies.
4. Run linting and tests.
5. Build the model-serving Docker image.
6. Start Minikube and apply the Kubernetes manifests.
7. Expose the Service locally and verify valid and invalid API requests.

Example commands from the repository root:

```powershell
Set-Location .\Phase-1-ML-Service
uv sync --frozen
uvx ruff check app.py train.py tests/
.\.venv\Scripts\python.exe -m pytest -v
docker build -t task2-ml-service:1.1 .
```

Return to the repository root before applying the Kubernetes manifests:

```powershell
Set-Location ..
minikube start
kubectl apply -f .\Phase-2-Kubernetes\namespace.yaml
kubectl apply -f .\Phase-2-Kubernetes\configmap.yaml
kubectl apply -f .\Phase-2-Kubernetes\depolyment.yaml
kubectl apply -f .\Phase-2-Kubernetes\service.yaml
```

For a local Minikube deployment, ensure the image referenced by the Deployment is available to the Minikube node. The project uses a local image workflow rather than requiring a remote container registry.

Reproducibility depends on the pinned dependency lockfile, model artifacts, configuration manifests, and compatible local tools. Commands and versions may need adjustment on a different operating system or cluster configuration.

## 4. Model-Drift Considerations

Model drift can occur when real-world input data changes over time or when the relationship between input data and expected predictions changes.

This project uses the scikit-learn 20 Newsgroups dataset. Its test accuracy does not guarantee equivalent performance on new or real-world data.

Considerations for future monitoring include:

- Track model versions and evaluation metrics for each release.
- Compare new evaluation results with the previous release before deployment.
- Monitor input-data patterns and prediction quality when representative labelled data becomes available.
- Investigate meaningful changes in data or performance and retrain when appropriate.
- Treat changes in prediction distributions as a signal for investigation, not proof of drift by themselves.

Automated drift detection and continuous production monitoring are not implemented in this project. They are future improvements, not completed features.

## 5. Limitations

- The project runs on a local Minikube cluster and does not require production infrastructure.
- The model is trained and evaluated on a limited benchmark dataset.
- The v1.1.0 test accuracy is slightly lower than v1.0.0.
- Automated production drift detection, scheduled retraining, monitoring dashboards, and alerting are not implemented.
- CI validates linting, tests, and image building; it does not automatically deploy the service to Kubernetes.
- Resource requests and limits are development settings and have not been established through production load testing.
- A separate bad-image failure simulation has not been confirmed.
- Local cluster and tunnel availability depend on the developer's machine and running tools.

## Conclusion

This project demonstrates a local ML prediction service deployed on Kubernetes with versioned model artifacts, configuration management, health probes, resource controls, automated CI checks, replica scaling, and recovery from a readiness-probe failure.

The workflow provides a reproducible foundation for experimenting with model releases and deployment reliability without requiring production infrastructure.