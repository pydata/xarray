from xarray.core.indexes import Index, PandasIndex


class ScalarIndex(Index):
    def __init__(self, value: int):
        self.value = value

    @classmethod
    def from_variables(cls, variables, *, options):
        var = next(iter(variables.values()))
        return cls(int(var.values))

    def equals(self, other, *, exclude=None):
        return isinstance(other, ScalarIndex) and other.value == self.value


class XYIndex(Index):
    def __init__(self, x: PandasIndex, y: PandasIndex):
        self.x: PandasIndex = x
        self.y: PandasIndex = y

    @classmethod
    def from_variables(cls, variables, *, options):
        return cls(
            x=PandasIndex.from_variables({"x": variables["x"]}, options=options),
            y=PandasIndex.from_variables({"y": variables["y"]}, options=options),
        )

    def equals(self, other, exclude=None):
        x_eq = True if self.x.dim in exclude else self.x.equals(other.x)
        y_eq = True if self.y.dim in exclude else self.y.equals(other.y)
        return x_eq and y_eq


class MultiCoordIndex(Index):
    def __init__(self, idx1, idx2):
        self.idx1 = idx1
        self.idx2 = idx2

    @classmethod
    def from_variables(cls, variables, *, options=None):
        idx1 = PandasIndex.from_variables({"x": variables["x"]}, options=options)
        idx2 = PandasIndex.from_variables({"y": variables["y"]}, options=options)

        return cls(idx1, idx2)

    def create_variables(self, variables=None):
        return {**self.idx1.create_variables(), **self.idx2.create_variables()}

    def isel(self, indexers):
        idx1 = self.idx1.isel({"x": indexers.get("x", slice(None))})
        idx2 = self.idx2.isel({"y": indexers.get("y", slice(None))})
        return MultiCoordIndex(idx1, idx2)
