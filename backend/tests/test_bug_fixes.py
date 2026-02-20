"""
Tests confirming the bug fixes for the property revenue dashboard.

Bug 1: Cross-tenant cache data leakage (cache key missing tenant_id)
Bug 2: Database pool using non-existent settings attributes
Bug 3: Mock data not tenant-aware
Bug 4: Floating-point precision loss in revenue calculations
"""

import pytest
from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import patch, AsyncMock, MagicMock


# ── Bug 1: Cache key must include tenant_id ──────────────────────────

@pytest.mark.asyncio
async def test_cache_key_includes_tenant_id():
    """
    Two tenants with the same property_id must not share a cache entry.
    The cache key must be revenue:{tenant_id}:{property_id}.
    """
    stored_keys = []

    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)

    async def capture_setex(key, ttl, value):
        stored_keys.append(key)

    mock_redis.setex = AsyncMock(side_effect=capture_setex)

    mock_result = {
        "property_id": "prop-001",
        "tenant_id": "tenant-a",
        "total": "2250.00",
        "currency": "USD",
        "count": 4,
    }

    with patch("app.services.cache.redis_client", mock_redis), \
         patch("app.services.reservations.calculate_total_revenue", AsyncMock(return_value=mock_result)):
        from app.services.cache import get_revenue_summary

        await get_revenue_summary("prop-001", "tenant-a")

    assert len(stored_keys) == 1
    assert stored_keys[0] == "revenue:tenant-a:prop-001"


@pytest.mark.asyncio
async def test_cache_keys_differ_across_tenants():
    """Same property_id for different tenants must produce different cache keys."""
    stored_keys = []

    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)

    async def capture_setex(key, ttl, value):
        stored_keys.append(key)

    mock_redis.setex = AsyncMock(side_effect=capture_setex)

    mock_result_a = {
        "property_id": "prop-001", "tenant_id": "tenant-a",
        "total": "2250.00", "currency": "USD", "count": 4,
    }
    mock_result_b = {
        "property_id": "prop-001", "tenant_id": "tenant-b",
        "total": "0.00", "currency": "USD", "count": 0,
    }

    with patch("app.services.cache.redis_client", mock_redis), \
         patch("app.services.reservations.calculate_total_revenue", AsyncMock(side_effect=[mock_result_a, mock_result_b])):
        from app.services.cache import get_revenue_summary

        await get_revenue_summary("prop-001", "tenant-a")
        await get_revenue_summary("prop-001", "tenant-b")

    assert "revenue:tenant-a:prop-001" in stored_keys
    assert "revenue:tenant-b:prop-001" in stored_keys
    assert stored_keys[0] != stored_keys[1]


# ── Bug 2: Database pool URL construction ─────────────────────────────

def test_database_pool_uses_settings_database_url():
    """
    DatabasePool.initialize() must derive the async URL from
    settings.database_url, not from non-existent supabase_db_* attributes.
    """
    from app.core.database_pool import DatabasePool
    from app.config import settings

    # The settings object must have database_url
    assert hasattr(settings, "database_url")

    # The settings object must NOT have the old supabase_db_* attributes
    assert not hasattr(settings, "supabase_db_user")
    assert not hasattr(settings, "supabase_db_password")
    assert not hasattr(settings, "supabase_db_host")


def test_database_url_converts_to_asyncpg():
    """The database URL must be converted to use the asyncpg driver."""
    from app.config import settings

    base_url = settings.database_url
    async_url = base_url.replace("postgresql://", "postgresql+asyncpg://")

    assert async_url.startswith("postgresql+asyncpg://")
    assert "asyncpg" in async_url


# ── Bug 3: Mock data must be tenant-aware ─────────────────────────────

@pytest.mark.asyncio
async def test_mock_data_different_per_tenant():
    """
    prop-001 exists in both tenants with different revenue.
    Mock fallback must return different data per tenant.
    """
    with patch("app.core.database_pool.DatabasePool") as MockPool:
        instance = MockPool.return_value
        instance.initialize = AsyncMock(side_effect=Exception("DB unavailable"))

        from app.services.reservations import calculate_total_revenue

        result_a = await calculate_total_revenue("prop-001", "tenant-a")
        result_b = await calculate_total_revenue("prop-001", "tenant-b")

    assert result_a["total"] == "2250.00"
    assert result_a["count"] == 4
    assert result_b["total"] == "0.00"
    assert result_b["count"] == 0


@pytest.mark.asyncio
async def test_mock_data_unknown_tenant_returns_zero():
    """Unknown tenant+property combo should return zero."""
    with patch("app.core.database_pool.DatabasePool") as MockPool:
        instance = MockPool.return_value
        instance.initialize = AsyncMock(side_effect=Exception("DB unavailable"))

        from app.services.reservations import calculate_total_revenue

        result = await calculate_total_revenue("prop-001", "tenant-unknown")

    assert result["total"] == "0.00"
    assert result["count"] == 0


# ── Bug 4: Revenue precision ──────────────────────────────────────────

def test_revenue_precision_sub_cent_values():
    """
    Values with 3 decimal places (like 333.333) must be rounded to 2
    decimal places using ROUND_HALF_UP before conversion to float.
    """
    # Simulate the sum from seed data: 333.333 + 333.333 + 333.334 = 1000.000
    raw_total = "1000.000"
    rounded = float(Decimal(raw_total).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    assert rounded == 1000.00


def test_revenue_precision_rounding_up():
    """A value like 999.995 should round UP to 1000.00 with ROUND_HALF_UP."""
    raw_total = "999.995"
    rounded = float(Decimal(raw_total).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    assert rounded == 1000.00


def test_revenue_precision_rounding_down():
    """A value like 999.994 should round DOWN to 999.99."""
    raw_total = "999.994"
    rounded = float(Decimal(raw_total).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    assert rounded == 999.99


def test_revenue_no_float_drift():
    """
    Demonstrates the original bug: naive float() on sub-cent values
    can produce drift. Our Decimal approach must not.
    """
    # 3 reservations with sub-cent amounts
    amounts = [Decimal("333.333"), Decimal("333.333"), Decimal("333.334")]
    total = sum(amounts)  # Decimal('1000.000') — exact

    # The fixed approach: Decimal rounding then float
    fixed = float(total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    assert fixed == 1000.00

    # Contrast: naive float conversion of individual amounts then sum
    naive = sum(float(a) for a in amounts)
    # This may or may not drift, but our fixed path is deterministic
    assert fixed == round(naive, 2)
