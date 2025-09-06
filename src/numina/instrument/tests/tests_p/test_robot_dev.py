import pytest

from numina.drps.tests.robot import RobotPos, Robot


def test_robot1(pattri_ins):
    dev = pattri_ins.get_device("robot")
    assert isinstance(dev, RobotPos)


def test_robot2(pattri_ins):
    base_dev = pattri_ins.get_device("robot")
    for dev in base_dev.children.values():
        assert isinstance(dev, Robot)


def test_robot3(pattri_ins):
    base_dev = pattri_ins.get_device("robot")
    for name, dev in base_dev.children.items():
        assert name.startswith("arm_")
        props = dev.get_property_names()
        assert "active" in props
        assert "angle" in props


def test_robot4(pattri_ins):
    base_dev = pattri_ins.get_device("robot")
    assert len(base_dev.children) == 7


def test_robot_hdr(pattri_ins, pattri_header):
    dev = pattri_ins.get_device("robot")
    hdr = pattri_header
    dev.configure_with_header(hdr)
    assert pattri_ins.get_device("robot.arm_1").angle == -0.3
    assert pattri_ins.get_device("robot.arm_2").active is True
    assert pattri_ins.get_device("robot.arm_3").active is False
    assert pattri_ins.get_device("robot.arm_3").angle == 0.4


def test_robot_hdr2(pattri_ins, pattri_header2):
    dev = pattri_ins.get_device("robot")
    dev.configure_with_header(pattri_header2)
    assert pattri_ins.get_device("robot.arm_1").angle == -0.3
    assert pattri_ins.get_device("robot.arm_2").active is False
    assert pattri_ins.get_device("robot.arm_3").active is False
    assert pattri_ins.get_device("robot.arm_3").angle == 0.4


def test_robot_active1():
    """If robot is not active, angle cant be changed"""
    dev = Robot("Robo", {}, 0, active=False)
    with pytest.warns(UserWarning):
        dev.angle = 2.4
    assert dev.angle == 0


def test_robot_active2():
    """If robot is not active, it can be configured"""
    dev = Robot("Robo", {}, 0, active=False)
    with dev.state.managed_configure():
        dev.angle = 2.4
    assert dev.angle == 2.4


def test_instrument2(pattri_ins):
    dev = pattri_ins.get_device("robot")
    for sub in dev.children:
        sub_dev = dev.children[sub]
        assert isinstance(sub_dev, Robot)
        assert sub_dev.name == sub

    assert isinstance(dev, RobotPos)
