"""
Constant velocity travel times
==============================

This example shows how to calculate the travel times of seismic waves in a
constant velocity model. We first define the model and the source and receiver
coordinates. We then calculate the travel times using the class 
:func:`~covseisnet.travel_times.TravelTimes`. Finally, we plot the travel times
on a map.
"""

import covseisnet as csn

# %%
# Create a constant velocity model
# --------------------------------
#
# We first create a constant velocity model with a velocity of 5 km/s. In order
# to do so, we simply need to define the geographical extent of the model, the
# resolution of the grid, and the velocity.

model = csn.velocity.ConstantVelocityModel(
    extent=(40, 41, 50, 51, 0, 20),
    shape=(20, 20, 20),
    velocity=3.5,
)

# %%
# Calculate the travel times between the sources and the receiver
# ---------------------------------------------------------------
#
# Each grid point of the model is considered as a source and the receiver is
# defined by the user. In the example below, the receiver is located at
# coordinates (40.7, 50.2, 0), somewhere in the model's domain. The travel
# times are calculated using the class
# :class:`~covseisnet.travel_times.TravelTimes`.
#
# We can then represent the travel times on a map using the method
# :func:`~covseisnet.plot.grid3d`.

# Calculate the travel times
traveltime = csn.travel_times.TravelTimes(
    model, receiver_coordinates=(40.7, 50.2, 0)
)

# Plot the traveltime grid
ax = csn.plot.grid3d(
    traveltime,
    cmap="RdPu",
    label="Travel time (s)",
    vmin=0,
)
