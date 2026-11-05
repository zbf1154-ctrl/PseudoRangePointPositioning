import pytest


def test_gui_module_importable():
    pytest.importorskip("tkinter")
    import spp.gui

    assert callable(spp.gui.main)
    assert hasattr(spp.gui, "SppApplication")


def test_gui_can_be_constructed():
    tkinter = pytest.importorskip("tkinter")
    try:
        root = tkinter.Tk()
    except tkinter.TclError:
        pytest.skip("当前环境没有可用的显示服务器")
    root.withdraw()
    try:
        from spp.gui import SppApplication

        app = SppApplication(root)
        root.update()
        assert app.status_var.get() == "就绪"
    finally:
        root.destroy()
