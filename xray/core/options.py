OPTIONS = {'display_width': 80}


class set_options(object):
    """ Set global state within controlled context
    """
    def __init__(self, **kwargs):
        self.old = OPTIONS.copy()
        OPTIONS.update(kwargs)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        OPTIONS.clear()
        OPTIONS.update(self.old)
