import pandas as pd
import xarray as xr

from . import TestCase, requires_netCDF4


class TestISODateParser(TestCase):
    def test_parse_iso8601(self):
        from xarray.core.netcdftimeindex import parse_iso8601

        def date_dict(year=None, month=None, day=None,
                      hour=None, minute=None, second=None):
            return dict(year=year, month=month, day=day, hour=hour,
                        minute=minute, second=second)

        for string, expected in [
            ('1999', date_dict(year='1999')),
            ('199901', date_dict(year='1999', month='01')),
            ('1999-01', date_dict(year='1999', month='01')),
            ('19990101', date_dict(year='1999', month='01', day='01')),
            ('1999-01-01', date_dict(year='1999', month='01', day='01')),
            ('19990101T12', date_dict(year='1999', month='01', day='01',
                                      hour='12')),
            ('1999-01-01T12', date_dict(year='1999', month='01', day='01',
                                        hour='12')),
            ('19990101T1234', date_dict(year='1999', month='01', day='01',
                                        hour='12', minute='34')),
            ('1999-01-01T12:34', date_dict(year='1999', month='01', day='01',
                                           hour='12', minute='34')),
            ('19990101T123456', date_dict(year='1999', month='01', day='01',
                                          hour='12', minute='34',
                                          second='56')),
            ('1999-01-01T12:34:56', date_dict(year='1999', month='01',
                                              day='01', hour='12', minute='34',
                                              second='56')),
            ('19990101T123456.78', date_dict(year='1999', month='01',
                                             day='01', hour='12', minute='34',
                                             second='56.78')),
            ('1999-01-01T12:34:56.78', date_dict(year='1999', month='01',
                                                 day='01', hour='12',
                                                 minute='34', second='56.78'))
        ]:
            result = parse_iso8601(string)
            self.assertEqual(result, expected)

            if '.' not in string:
                with self.assertRaises(ValueError):
                    parse_iso8601(string + '3')


@requires_netCDF4
class NetCDFTimeIndexTests(object):
    feb_days = 28
    dec_days = 31

    def set_date_type(self):
        self.date_type = None

    def setUp(self):
        from xarray.core.netcdftimeindex import NetCDFTimeIndex
        self.set_date_type()
        dates = [self.date_type(1, 1, 1), self.date_type(1, 2, 1),
                 self.date_type(2, 1, 1), self.date_type(2, 2, 1)]
        self.index = NetCDFTimeIndex(dates)
        self.da = xr.DataArray([1, 2, 3, 4], coords=[self.index],
                               dims=['time'])
        self.series = pd.Series([1, 2, 3, 4], index=self.index)
        self.df = pd.DataFrame([1, 2, 3, 4], index=self.index)

    def tearDown(self):
        pass

    def test_netcdftimeindex_field_accessors(self):
        for field, expected in [
                ('year', [1, 1, 2, 2]),
                ('month', [1, 2, 1, 2]),
                ('day', [1, 1, 1, 1]),
                ('hour', [0, 0, 0, 0]),
                ('minute', [0, 0, 0, 0]),
                ('second', [0, 0, 0, 0]),
                ('microsecond', [0, 0, 0, 0])
        ]:
            result = getattr(self.index, field)
            self.assertArrayEqual(result, expected)

    def test_parse_iso8601_with_reso(self):
        from xarray.core.netcdftimeindex import _parse_iso8601_with_reso

        for string, (expected_date, expected_reso) in [
                ('1999', (self.date_type(1999, 1, 1), 'year')),
                ('199902', (self.date_type(1999, 2, 1), 'month')),
                ('19990202', (self.date_type(1999, 2, 2), 'day')),
                ('19990202T01', (self.date_type(1999, 2, 2, 1), 'hour')),
                ('19990202T0101', (self.date_type(1999, 2, 2, 1, 1),
                                   'minute')),
                ('19990202T010156', (self.date_type(1999, 2, 2, 1, 1, 56),
                                     'second'))
        ]:
            result_date, result_reso = _parse_iso8601_with_reso(
                self.date_type, string)
            self.assertEqual(result_date, expected_date)
            self.assertEqual(result_reso, expected_reso)

    def test_parsed_string_to_bounds(self):
        from xarray.core.netcdftimeindex import _parsed_string_to_bounds
        parsed = self.date_type(2, 2, 10, 6, 2, 8, 1)

        for resolution, (expected_start, expected_end) in [
            ('year', (self.date_type(2, 1, 1),
                      self.date_type(2, 12, self.dec_days, 23, 59, 59,
                                     999999))),
            ('month', (self.date_type(2, 2, 1),
                       self.date_type(2, 2, self.feb_days, 23, 59, 59,
                                      999999))),
            ('day', (self.date_type(2, 2, 10),
                     self.date_type(2, 2, 10, 23, 59, 59, 999999))),
            ('hour', (self.date_type(2, 2, 10, 6),
                      self.date_type(2, 2, 10, 6, 59, 59, 999999))),
            ('minute', (self.date_type(2, 2, 10, 6, 2),
                        self.date_type(2, 2, 10, 6, 2, 59, 999999))),
            ('second', (self.date_type(2, 2, 10, 6, 2, 8),
                        self.date_type(2, 2, 10, 6, 2, 8, 999999)))
        ]:
            result_start, result_end = _parsed_string_to_bounds(
                self.date_type, resolution, parsed)
            self.assertEqual(result_start, expected_start)
            self.assertEqual(result_end, expected_end)

        # Test special case for monthly resolution and parsed date in December
        parsed = self.date_type(2, 12, 1)
        expected_start = self.date_type(2, 12, 1)
        expected_end = self.date_type(2, 12, self.dec_days, 23, 59, 59, 999999)
        result_start, result_end = _parsed_string_to_bounds(
            self.date_type, 'month', parsed)
        self.assertEqual(result_start, expected_start)
        self.assertEqual(result_end, expected_end)

    def test_get_loc(self):
        result = self.index.get_loc('0001')
        expected = [0, 1]
        self.assertArrayEqual(result, expected)

        result = self.index.get_loc(self.date_type(1, 2, 1))
        expected = 1
        self.assertEqual(result, expected)

        result = self.index.get_loc('0001-02-01')
        expected = 1
        self.assertEqual(result, expected)

    def test_get_slice_bound(self):
        for kind in ['loc', 'getitem']:
            result = self.index.get_slice_bound('0001', 'left', kind)
            expected = 0
            self.assertEqual(result, expected)

            result = self.index.get_slice_bound('0001', 'right', kind)
            expected = 2
            self.assertEqual(result, expected)

            result = self.index.get_slice_bound(
                self.date_type(1, 3, 1), 'left', kind)
            expected = 2
            self.assertEqual(result, expected)

            result = self.index.get_slice_bound(
                self.date_type(1, 3, 1), 'right', kind)
            expected = 2
            self.assertEqual(result, expected)

    def test_date_type_property(self):
        self.assertEqual(self.index.date_type, self.date_type)

    def test_contains(self):
        assert '0001' in self.index
        assert '0003' not in self.index
        assert self.date_type(1, 1, 1) in self.index
        assert self.date_type(3, 1, 1) not in self.index

    def test_groupby(self):
        result = self.da.groupby('time.month').sum('time')
        expected = xr.DataArray([4, 6], coords=[[1, 2]], dims=['month'])
        self.assertDataArrayIdentical(result, expected)

    def test_sel(self):
        expected = xr.DataArray([1, 2], coords=[self.index[:2]], dims=['time'])
        for result in [
            self.da.sel(time='0001'),
            self.da.sel(time=slice('0001-01-01', '0001-12-30')),
            self.da.sel(time=slice(self.date_type(1, 1, 1),
                                   self.date_type(1, 12, 30))),
            self.da.sel(time=[self.date_type(1, 1, 1),
                              self.date_type(1, 2, 1)]),
            self.da.sel(time=[True, True, False, False])
        ]:
            self.assertDataArrayIdentical(result, expected)

        expected = xr.DataArray(1).assign_coords(time=self.index[0])
        for result in [
            self.da.sel(time=self.date_type(1, 1, 1)),
            self.da.sel(time='0001-01-01')
        ]:
            self.assertDataArrayIdentical(result, expected)

    def test_isel(self):
        expected = xr.DataArray(1).assign_coords(time=self.index[0])
        result = self.da.isel(time=0)
        self.assertDataArrayIdentical(result, expected)

        expected = xr.DataArray([1, 2], coords=[self.index[:2]], dims=['time'])
        result = self.da.isel(time=[0, 1])
        self.assertDataArrayIdentical(result, expected)

    def test_indexing_in_series(self):
        # Note that integer-based indexing outside of iloc does not work
        # using the simplified get_value method (for now).
        expected = 1
        for result in [
            # self.series[0],
            self.series[self.date_type(1, 1, 1)],
            self.series['0001-01-01'],
            self.series.loc['0001-01-01'],
            self.series.loc[self.date_type(1, 1, 1)],
            self.series.iloc[0]
        ]:
            self.assertEqual(result, expected)
            self.assertEqual(result, expected)

        expected = pd.Series([1, 2], index=self.index[:2])
        for result in [
            # self.series[:2],
            self.series['0001'],
            self.series['0001-01-01':'0001-12-30'],
            self.series[self.date_type(1, 1, 1):self.date_type(1, 12, 30)],
            self.series.loc[self.date_type(1, 1, 1):self.date_type(1, 12, 30)],
            self.series.loc['0001'],
            self.series.loc['0001-01-01':'0001-12-30'],
            self.series.loc[:'0001-12-30'],
            self.series.iloc[:2]
        ]:
            pd.util.testing.assert_series_equal(result, expected)

    def test_indexing_in_dataframe(self):
        expected = pd.Series([1], name=self.index[0])
        for result in [
            self.df.loc['0001-01-01'],
            self.df.loc[self.date_type(1, 1, 1)],
            self.df.iloc[0]
        ]:
            pd.util.testing.assert_series_equal(result, expected)

        expected = pd.DataFrame([1, 2], index=self.index[:2])
        for result in [
            self.df.loc['0001'],
            self.df.loc['0001-01-01':'0001-12-30'],
            self.df.loc[:'0001-12-30'],
            self.df.loc[self.date_type(1, 1, 1):self.date_type(1, 12, 30)],
            self.df.iloc[:2]
        ]:
            pd.util.testing.assert_frame_equal(result, expected)


class DatetimeJulianTestCase(NetCDFTimeIndexTests, TestCase):
    def set_date_type(self):
        from netcdftime import DatetimeJulian
        self.date_type = DatetimeJulian


class DatetimeGregorianTestCase(NetCDFTimeIndexTests, TestCase):
    def set_date_type(self):
        from netcdftime import DatetimeGregorian
        self.date_type = DatetimeGregorian


class DatetimeProlepticGregorianTestCase(NetCDFTimeIndexTests, TestCase):
    def set_date_type(self):
        from netcdftime import DatetimeProlepticGregorian
        self.date_type = DatetimeProlepticGregorian


class DatetimeNoLeapTestCase(NetCDFTimeIndexTests, TestCase):
    def set_date_type(self):
        from netcdftime import DatetimeNoLeap
        self.date_type = DatetimeNoLeap


class DatetimeAllLeapTestCase(NetCDFTimeIndexTests, TestCase):
    feb_days = 29

    def set_date_type(self):
        from netcdftime import DatetimeAllLeap
        self.date_type = DatetimeAllLeap


class Datetime360DayTestCase(NetCDFTimeIndexTests, TestCase):
    feb_days = 30
    dec_days = 30

    def set_date_type(self):
        from netcdftime import Datetime360Day
        self.date_type = Datetime360Day
