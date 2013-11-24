scidata
=======

Objects for holding self describing scientific data in python.  The goal of this project is to
provide a Common Data Model (http://www.unidata.ucar.edu/software/thredds/current/netcdf-java/CDM/)
allowing users to read write and manipulate netcdf-like data without worrying about where the data 
source lives. A dataset that is too large to fit in memory, served from an OpenDAP server, streamed 
or stored as NetCDF3, NetCDF4, grib (?), HDF5 and others can all be inspected and manipulated using 
the same methods.

Of course there are already several packages in python that offer similar functionality (netCDF4, 
scipy.io, pupynere, iris, ... ) but each of those packages have their own shortcomings:

netCDF4
    Doesn't allow streaming.  If you want to create a new object it needs to live on disk.
scipy.io / pupynere
    Only works with NetCDF3 and doesn't support DAP making it difficult to work with large datasets.
iris
    is REALLY close to what this project will provide, but iris strays further from the CDM,
    than I would like. (if you read then write a netcdf file using iris all global attributes 
    are pushed down to variable level attributes.
