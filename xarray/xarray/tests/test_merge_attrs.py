import pytest
import xarray as xr

def test_merge_override_attrs_copy():
    """Test that merge with combine_attrs='override' creates a copy of attrs"""
    # Create two datasets with different attrs
    xds1 = xr.Dataset(attrs={'a': 'b', 'b': 'c'})
    xds2 = xr.Dataset(attrs={'a': 'c', 'd': 'e'})
    
    # Store original attrs values
    original_a1 = xds1.attrs['a']
    original_a2 = xds2.attrs['a']
    
    # Merge with override
    xds3 = xr.merge([xds1, xds2], combine_attrs='override')
    
    # Verify initial merge result
    assert xds3.attrs['a'] == 'b'  # Should take value from first dataset
    assert xds3.attrs['b'] == 'c'
    assert xds3.attrs['d'] == 'e'
    
    # Modify the merged dataset's attrs
    xds3.attrs['a'] = 'd'
    
    # Verify that original datasets are unchanged
    assert xds1.attrs['a'] == original_a1, 'Original dataset attrs were modified'
    assert xds2.attrs['a'] == original_a2, 'Second dataset attrs were modified'
    assert xds1.attrs['a'] == 'b', 'First dataset attr value changed'
    
    # Verify that attrs are independent (not the same object)
    assert xds3.attrs is not xds1.attrs, 'Attrs are not independent copies'
    
    # Test with empty inputs
    assert xr.core.merge.merge_attrs([], 'override') == {}
    assert xr.core.merge.merge_attrs([{}], 'override') == {}
    
    # Test with single input
    single_attrs = {'x': 'y', 'z': 'w'}
    result = xr.core.merge.merge_attrs([single_attrs], 'override')
    assert result == single_attrs
    assert result is not single_attrs, 'Single input should still create a copy'
    result['x'] = 'modified'
    assert single_attrs['x'] == 'y', 'Original single attrs were modified'

