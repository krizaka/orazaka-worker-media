"""How this worker reaches the broker — one place, for the consumer and for every event it publishes.

The worker used to build ``pika.ConnectionParameters(host, port)`` in two modules and nothing
else, which means it logged in as ``guest`` — a user RabbitMQ only accepts from localhost. That was
true on the Mac, where the broker is a Docker container on the same machine, and false everywhere
else: the GPU droplet reaches the dev broker through an authenticated tunnel, with the broker's own
credentials. The connection is therefore declared by the deployment, never assumed.

The variable names are Spring's (``SPRING_RABBITMQ_*``) on purpose: the Java services read the same
ones, so one environment file describes the broker for every process of the platform.
"""
from __future__ import annotations

import os
import ssl

import pika

#: Long enough for the tunnel's keep-alives, short enough that a dead link is noticed.
DEFAULT_HEARTBEAT_SECONDS = 60


def connection_parameters(env: dict | None = None) -> pika.ConnectionParameters:
    """The broker this process talks to, as the deployment declared it.

    Args:
        env: the environment to read (tests pass a dict); ``os.environ`` by default.

    Returns:
        The parameters for ``pika.BlockingConnection``: host, port, virtual host, credentials,
        heartbeat and — when ``SPRING_RABBITMQ_SSL_ENABLED`` is ``true`` — TLS with certificate and
        host-name verification (never an unverified context).
    """
    source = os.environ if env is None else env

    def read(name: str, default: str) -> str:
        value = source.get(name)
        return value if value else default

    host = read("SPRING_RABBITMQ_HOST", "localhost")
    try:
        port = int(read("SPRING_RABBITMQ_PORT", "5672"))
    except ValueError:
        port = 5672
    try:
        heartbeat = int(read("SPRING_RABBITMQ_HEARTBEAT", str(DEFAULT_HEARTBEAT_SECONDS)))
    except ValueError:
        heartbeat = DEFAULT_HEARTBEAT_SECONDS
    credentials = pika.PlainCredentials(
        read("SPRING_RABBITMQ_USERNAME", "guest"), read("SPRING_RABBITMQ_PASSWORD", "guest")
    )
    options = {
        "host": host,
        "port": port,
        "virtual_host": read("SPRING_RABBITMQ_VIRTUAL_HOST", "/"),
        "credentials": credentials,
        "heartbeat": heartbeat,
        "blocked_connection_timeout": 300,
    }
    if read("SPRING_RABBITMQ_SSL_ENABLED", "false").lower() == "true":
        # A private CA (the dev broker's) is trusted when the deployment names its file; the name the
        # certificate is checked against may differ from the address dialled — the GPU host reaches
        # broker-dev.orazaka.com through a local tunnel end (SPRING_RABBITMQ_SSL_SERVER_NAME).
        cafile = source.get("SPRING_RABBITMQ_SSL_CA_FILE") or None
        context = ssl.create_default_context(cafile=cafile)
        server_name = read("SPRING_RABBITMQ_SSL_SERVER_NAME", host)
        options["ssl_options"] = pika.SSLOptions(context, server_hostname=server_name)
    return pika.ConnectionParameters(**options)


def describe(params: pika.ConnectionParameters) -> str:
    """A log line for the connection — host, port and user, never the password."""
    user = getattr(params.credentials, "username", "?")
    scheme = "amqps" if params.ssl_options else "amqp"
    return f"{scheme}://{user}@{params.host}:{params.port}{params.virtual_host}"
