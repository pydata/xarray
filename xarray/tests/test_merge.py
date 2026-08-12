import pytest
import xarray as xr
from xarray.core.merge import merge_attrs

def test_merge_override_attrs_copy():
    # Test that override mode creates a copy of attrs, not a reference
    attrs1 = {'a': 'b', 'b': 'c'}
    attrs2 = {'a': 'c', 'd': 'e'}
    
    # Test with datasets
    xds1 = xr.Dataset(attrs=attrs1)
    xds2 = xr.Dataset(attrs=attrs2)
    
    xds3 = xr.merge([xds1, xds2], combine_attrs='override')
    
    # Modify the merged dataset's attrs
    original_a1 = xds1.attrs['a']
    xds3.attrs['a'] = 'd'
    
    # The original dataset should not be affected
    assert xds1.attrs['a'] == original_a1, "Original dataset attrs were modified"
    assert xds3.attrs['a'] == 'd', "Merged dataset attrs not properly set"
    
    # The attrs should be independent objects
    assert xds3.attrs is not xds1.attrs, "Attrs are not independent copies"
    
    # Test the merge_attrs function directly
    merged_attrs = merge_attrs([attrs1, attrs2], 'override')
    assert merged_attrs == attrs1
    assert merged_attrs is not attrs1, "merge_attrs should return a copy, not a reference"
    
    # Modify the merged attrs and ensure original is unchanged
    merged_attrs['a'] = 'modified'
    assert attrs1['a'] == 'b', "Original attrs were modified by changes to merged attrs"

def test_merge_override_attrs_empty():
    # Test override with empty list
    result = merge_attrs([], 'override')
    assert result is None
    
    # Test override with single empty dict
    result = merge_attrs([{}], 'override')
    assert result == {}
    assert isinstance(result, dict)
