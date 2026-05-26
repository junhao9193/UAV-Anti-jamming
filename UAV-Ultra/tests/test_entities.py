"""Stage 1 实体契约测试。

这些测试只锁定旧 `envs/entities.py` 已经存在的公开属性和初始化语义。它们不测试
运动、链路或奖励逻辑，因为那些职责会在后续 envs 拆分阶段迁移。
"""

from src.entities import Jammer, RP, UAV


def test_uav_initializes_legacy_public_attributes():
    """UAV 应保留旧实现依赖的公开属性名和初始赋值关系。"""

    position = [1.0, 2.0, 3.0]
    uav = UAV(position, start_direction=0.5, start_velocity=12.0, start_p=0.2)

    assert uav.position is position
    assert uav.direction == 0.5
    assert uav.velocity == 12.0
    assert uav.p == 0.2
    assert uav.mean_velocity == 12.0
    assert uav.mean_direction == 0.5
    assert uav.mean_p == 0.2
    assert uav.destinations == []
    assert uav.connections == []


def test_uav_mutable_lists_are_per_instance():
    """目的节点和连接记录必须是实例私有列表，避免多个实体共享状态。"""

    first = UAV([0, 0, 0], 0.0, 1.0, 0.0)
    second = UAV([1, 1, 1], 0.0, 1.0, 0.0)

    first.destinations.append("rp-0")
    first.connections.append("link-0")

    assert second.destinations == []
    assert second.connections == []


def test_jammer_initializes_legacy_public_attributes():
    """Jammer 应保留旧实现的公开属性和 mean_* 初始化规则。"""

    position = [4.0, 5.0, 6.0]
    jammer = Jammer(position, start_direction=1.5, velocity=8.0, start_p=0.4)

    assert jammer.position is position
    assert jammer.direction == 1.5
    assert jammer.velocity == 8.0
    assert jammer.p == 0.4
    assert jammer.mean_velocity == 8.0
    assert jammer.mean_direction == 1.5
    assert jammer.mean_p == 0.4


def test_rp_initializes_legacy_public_attributes():
    """RP 只保存位置和连接记录，保持旧实现的轻量数据容器语义。"""

    position = [7.0, 8.0, 9.0]
    rp = RP(position)

    assert rp.position is position
    assert rp.connections == []
