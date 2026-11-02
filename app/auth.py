import os

from clerk_backend_api import Clerk
from clerk_backend_api.security.types import AuthenticateRequestOptions
from fastapi import HTTPException, Request


def require_auth(request: Request) -> dict:
    secret_key = os.getenv("CLERK_SECRET_KEY")

    if not secret_key:
        raise HTTPException(
            status_code=500,
            detail="Authentication is not configured.",
        )

    authorized_parties = [
        value.strip()
        for value in os.getenv("CLERK_AUTHORIZED_PARTIES", "").split(",")
        if value.strip()
    ]

    if not authorized_parties:
        raise HTTPException(
            status_code=500,
            detail="Authorized authentication origins are not configured.",
        )

    sdk = Clerk(bearer_auth=secret_key)

    state = sdk.authenticate_request(
        request,
        AuthenticateRequestOptions(
            authorized_parties=authorized_parties,
            accepts_token=["session_token"],
        ),
    )

    if not state.is_signed_in:
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    expected_org_id = os.getenv("CLERK_TEST_ORGANIZATION_ID")
    actual_org_id = state.payload.get("org_id")

    if not expected_org_id:
        raise HTTPException(
            status_code=500,
            detail="Test organization is not configured.",
        )

    if actual_org_id != expected_org_id:
        raise HTTPException(
            status_code=403,
            detail="Organization access denied.",
        )

    return state.payload