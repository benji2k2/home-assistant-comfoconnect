"""Session handling: a failed setup must close its session, and a refused session must recover."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

# These tests set up the integration in a real Home Assistant instance. They need
# pytest-homeassistant-custom-component (and asyncio_mode = "auto"), which the dev
# dependencies don't include yet, so skip them where it isn't installed.
pytest.importorskip("pytest_homeassistant_custom_component")

from aiocomfoconnect.exceptions import AioComfoConnectNotConnected, ComfoConnectNotAllowed, ComfoConnectRmiError
from custom_components.comfoconnect import KEEP_ALIVE_INTERVAL, ComfoConnectBridge
from custom_components.comfoconnect.const import CONF_LOCAL_UUID, CONF_UUID, DOMAIN
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.const import CONF_HOST
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_fire_time_changed

BRIDGE = "custom_components.comfoconnect.ComfoConnectBridge"


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Load the integration from custom_components."""
    return


@pytest.fixture(name="entry")
def entry_fixture(hass: HomeAssistant) -> MockConfigEntry:
    """A config entry for one bridge."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="192.0.2.10",
        data={CONF_HOST: "192.0.2.10", CONF_UUID: "00" * 16, CONF_LOCAL_UUID: "11" * 16},
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture(name="bridge")
def bridge_fixture():
    """The bridge, without a network: every call the integration makes is mocked."""
    with (
        patch(f"{BRIDGE}.connect", new_callable=AsyncMock) as connect,
        patch(f"{BRIDGE}.disconnect", new_callable=AsyncMock) as disconnect,
        patch(
            f"{BRIDGE}.cmd_version_request",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(serialNumber="DEMO0001", gatewayVersion=0),
        ),
        patch(f"{BRIDGE}.get_property", new_callable=AsyncMock, return_value="ComfoAir Q350") as get_property,
        patch(f"{BRIDGE}.cmd_time_request", new_callable=AsyncMock) as time_request,
        patch("custom_components.comfoconnect.version_decode", return_value="1.0.0"),
        patch(
            "homeassistant.config_entries.ConfigEntries.async_forward_entry_setups",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "homeassistant.config_entries.ConfigEntries.async_unload_platforms",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        yield SimpleNamespace(connect=connect, disconnect=disconnect, get_property=get_property, time_request=time_request)


async def keepalive(hass: HomeAssistant) -> None:
    """Let one keepalive interval pass."""
    async_fire_time_changed(hass, dt_util.utcnow() + KEEP_ALIVE_INTERVAL + timedelta(seconds=1))
    await hass.async_block_till_done()


async def test_failed_device_information_closes_the_session(hass: HomeAssistant, entry, bridge) -> None:
    """
    Seen after a restart: the first property read fails with an RMI error.

    The session was already started, so it must be closed again, and the entry must retry
    instead of staying in setup_error. Before, the unreferenced session kept reconnecting and
    took the session over from the next attempt.
    """
    bridge.get_property.side_effect = ComfoConnectRmiError("RMI_ERROR 45")

    assert not await hass.config_entries.async_setup(entry.entry_id)

    assert entry.state is ConfigEntryState.SETUP_RETRY
    bridge.disconnect.assert_awaited_once()
    assert entry.entry_id not in hass.data.get(DOMAIN, {})


async def test_setup_succeeds(hass: HomeAssistant, entry, bridge) -> None:
    """The normal path is unchanged."""
    assert await hass.config_entries.async_setup(entry.entry_id)

    assert entry.state is ConfigEntryState.LOADED
    bridge.disconnect.assert_not_awaited()
    assert isinstance(hass.data[DOMAIN][entry.entry_id], ComfoConnectBridge)


async def test_refused_session_starts_a_new_one(hass: HomeAssistant, entry, bridge) -> None:
    """The bridge answers NOT_ALLOWED while the connection is still open: start a new session."""
    assert await hass.config_entries.async_setup(entry.entry_id)
    ccb = hass.data[DOMAIN][entry.entry_id]
    bridge.connect.reset_mock()
    # Refused once, then the new session answers.
    bridge.time_request.side_effect = [ComfoConnectNotAllowed("NOT_ALLOWED"), None]

    await keepalive(hass)

    bridge.disconnect.assert_awaited_once()
    bridge.connect.assert_awaited_once_with(entry.data[CONF_LOCAL_UUID])
    assert ccb.is_available


async def test_refused_new_session_asks_for_reauthentication(hass: HomeAssistant, entry, bridge) -> None:
    """When the bridge refuses the new session as well, we are no longer registered."""
    assert await hass.config_entries.async_setup(entry.entry_id)
    ccb = hass.data[DOMAIN][entry.entry_id]
    bridge.time_request.side_effect = ComfoConnectNotAllowed("NOT_ALLOWED")
    bridge.connect.side_effect = ComfoConnectNotAllowed("NOT_ALLOWED")

    with patch.object(ConfigEntry, "async_start_reauth") as start_reauth:
        await keepalive(hass)

    start_reauth.assert_called_once()
    assert not ccb.is_available


async def test_working_keepalive_does_not_reconnect(hass: HomeAssistant, entry, bridge) -> None:
    """A keepalive that gets an answer leaves the session alone."""
    assert await hass.config_entries.async_setup(entry.entry_id)
    bridge.connect.reset_mock()

    await keepalive(hass)

    bridge.time_request.assert_awaited()
    bridge.connect.assert_not_awaited()
    bridge.disconnect.assert_not_awaited()


async def settle() -> None:
    """Let the event loop run the scheduled keepalive as far as it gets."""
    for _ in range(20):
        await asyncio.sleep(0)


def fire_keepalive(hass: HomeAssistant, times: int = 1) -> None:
    """Move the clock on by `times` keepalive intervals."""
    async_fire_time_changed(hass, dt_util.utcnow() + times * (KEEP_ALIVE_INTERVAL + timedelta(seconds=1)))


async def test_available_only_once_the_new_session_answers(hass: HomeAssistant, entry, bridge) -> None:
    """connect() returns at once while the library is still reconnecting: stay unavailable."""
    assert await hass.config_entries.async_setup(entry.entry_id)
    ccb = hass.data[DOMAIN][entry.entry_id]
    bridge.time_request.side_effect = AioComfoConnectNotConnected("not connected")

    with patch.object(ConfigEntry, "async_start_reauth") as start_reauth:
        await keepalive(hass)

    bridge.connect.assert_awaited()
    assert not ccb.is_available
    start_reauth.assert_not_called()


async def test_overlapping_keepalive_is_skipped(hass: HomeAssistant, entry, bridge) -> None:
    """A keepalive that fires while the previous one is still reconnecting does nothing."""
    assert await hass.config_entries.async_setup(entry.entry_id)
    bridge.connect.reset_mock()
    release = asyncio.Event()

    async def slow_connect(*args):
        await release.wait()

    bridge.connect.side_effect = slow_connect
    bridge.time_request.side_effect = [AioComfoConnectNotConnected("not connected"), None, None]

    fire_keepalive(hass)
    await settle()
    assert bridge.connect.await_count == 1, "the first keepalive is reconnecting"

    fire_keepalive(hass, 2)
    await settle()
    assert bridge.connect.await_count == 1, "the second one must not start another connect"
    assert bridge.time_request.await_count == 1, "nor send a keepalive in between"

    release.set()
    await hass.async_block_till_done()
    assert hass.data[DOMAIN][entry.entry_id].is_available


async def test_unload_waits_for_a_running_keepalive(hass: HomeAssistant, entry, bridge) -> None:
    """A keepalive that is still running when the entry unloads must not open a new session."""
    assert await hass.config_entries.async_setup(entry.entry_id)
    bridge.connect.reset_mock()
    release = asyncio.Event()

    async def hanging_time_request():
        await release.wait()
        raise AioComfoConnectNotConnected("not connected")

    bridge.time_request.side_effect = hanging_time_request

    fire_keepalive(hass)
    await settle()
    unload = hass.async_create_task(hass.config_entries.async_unload(entry.entry_id))
    await settle()
    bridge.disconnect.assert_not_awaited()  # waits for the keepalive

    release.set()
    assert await unload

    bridge.connect.assert_not_awaited()
    bridge.disconnect.assert_awaited_once()
    assert entry.state is ConfigEntryState.NOT_LOADED
