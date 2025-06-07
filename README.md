# Kubernetes Deployments for Ollama Chat Interface

This project showcases Kubernetes deployments for an Ollama Chat Interface. While the application itself demonstrates communication between a Python app and Ollama, as well as with online models, the primary focus is on the Kubernetes deployment aspects.

## Key Features

-   Kubernetes deployments using Helm charts
-   Rancher Fleet configurations for managing deployments
-   Support for both upstream and SUSE environments

## Architecture

The project includes the following main directories:

-   `charts/`: Helm charts for deploying the application on Kubernetes.
    -   `upstream/`: Helm charts for upstream Kubernetes environments.
    -   `suse/`: Helm charts for SUSE Rancher environments.
-   `fleet/`: Rancher Fleet configurations for managing deployments.
    -   `upstream/`: Fleet configurations for upstream Kubernetes environments.
    -   `suse/`: Fleet configurations for SUSE Rancher environments.
-   `app/`: Contains the Python application code for the Ollama Chat Interface.
    -   `main.py`: The main Python file that defines the Gradio interface and chat logic.
    -   `requirements.txt`: The list of Python packages required to run the application.

## Deployment

This project offers three different ways to deploy the Ollama Chat Interface on Kubernetes: using Helm, Rancher Fleet, or directly through the Rancher UI.

### Prerequisites

-   A Kubernetes cluster
-   Helm installed
-   Rancher Fleet installed (optional)
-   Rancher UI access (optional)

### Deployment Methods

#### 1. Helm

1.  Install the application using Helm:

    ```bash
    helm install ollama-chat charts/upstream
    ```

    or

    ```bash
    helm install ollama-chat charts/suse
    ```

#### 2. Rancher Fleet

1.  Deploy the application using Rancher Fleet:

    ```bash
    fleet apply -f fleet/upstream
    ```

    or

    ```bash
    fleet apply -f fleet/suse
    ```

#### 3. Rancher UI

1.  Add this repository as an App in the Rancher UI.
2.  Configure the deployment settings as needed.
3.  Deploy the application.

## Configuration

The Kubernetes deployments can be configured using the values.yaml files in the `charts/` directories.

## Contributing

Contributions are welcome! Please submit a pull request with your changes.

## License

[Specify the license here]
