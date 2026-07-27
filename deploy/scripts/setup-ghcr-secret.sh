#!/usr/bin/env bash
# Create/update ghcr-secret in a namespace after verifying GHCR pull works.
set -euo pipefail

NAMESPACE="${1:-ai-md}"
USERNAME="${GHCR_USERNAME:-nicholas-stelle}"
SECRET_NAME="${GHCR_SECRET_NAME:-ghcr-secret}"
IMAGE="${GHCR_IMAGE:-ghcr.io/rose-stl-lab/sindyffuse:latest}"

if [[ -z "${GHCR_PAT:-}" ]]; then
  read -r -s -p "GitHub PAT (classic read:packages, SSO authorized for Rose-STL-Lab): " GHCR_PAT
  echo
fi

if [[ -z "$GHCR_PAT" ]]; then
  echo "error: empty PAT" >&2
  exit 1
fi

echo "Testing GHCR login and pull for ${IMAGE} ..."
if ! echo "$GHCR_PAT" | podman login ghcr.io -u "$USERNAME" --password-stdin >/dev/null; then
  echo "error: podman login to ghcr.io failed" >&2
  exit 1
fi

if ! podman pull "$IMAGE" >/dev/null; then
  echo "error: podman pull failed — fix PAT permissions before creating the k8s secret:" >&2
  echo "  - Use a classic PAT with read:packages (fine-grained often fails for GHCR org packages)" >&2
  echo "  - Settings → PAT → Configure SSO → Authorize Rose-STL-Lab" >&2
  echo "  - Packages → sindyffuse → Manage access → grant your user Read" >&2
  exit 1
fi

echo "GHCR pull OK. Writing ${SECRET_NAME} in namespace ${NAMESPACE} ..."
kubectl delete secret "$SECRET_NAME" -n "$NAMESPACE" --ignore-not-found
kubectl create secret docker-registry "$SECRET_NAME" -n "$NAMESPACE" \
  --docker-server=ghcr.io \
  --docker-username="$USERNAME" \
  --docker-password="$GHCR_PAT" \
  --docker-email="${USERNAME}@users.noreply.github.com"

echo "Done. Restart jobs so pods pick up the secret:"
echo "  kubectl delete job sindyffuse-preprocess-mint -n ${NAMESPACE} --ignore-not-found"
echo "  kubectl apply -k deploy/jobs/preprocess-mint/workers -n ${NAMESPACE}"
