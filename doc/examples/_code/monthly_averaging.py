def year_month(xray_obj):
    """Given an xray object with a 'time' coordinate, return an DataArray
    with values given by the first date of the month in which each time
    falls.
    """
    time = xray_obj.coords['time']
    values = time.to_index().to_period('M').to_timestamp()
    return xray.DataArray(values, [time], name='year_month')
