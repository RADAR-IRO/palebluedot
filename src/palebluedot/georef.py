from pyproj import Geod
import rasterio
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint
from datetime import timedelta


def position_at(satellite, time):
    lon, lat, _ = satellite.get_position(time).position_llh
    return lat, lon


def compute(image, satellite, start, end, span, desc=None):
    """
    Add georeferencing to an image captured by a weather satellite.

    :param image: path to the original image file.
    :param satellite: satellite position predictor.
    :param start: capture start date.
    :param end: capture end date.
    :param span: span of the image to the left and right (in meters).
    :param desc: human-readable data description.
    """
    start_loc = position_at(satellite, start)
    after_start_loc = position_at(satellite, start + timedelta(seconds=5))
    before_end_loc = position_at(satellite, end - timedelta(seconds=5))
    end_loc = position_at(satellite, end)

    geod = Geod(ellps="WGS84")

    start_azimuth, *_ = geod.inv(*start_loc, *after_start_loc)
    *start_left_loc, _ = geod.fwd(*start_loc, start_azimuth + 90, span)
    *start_right_loc, _ = geod.fwd(*start_loc, start_azimuth - 90, span)

    end_azimuth, *_ = geod.inv(*before_end_loc, *end_loc)
    *end_left_loc, _ = geod.fwd(*end_loc, end_azimuth + 90, span)
    *end_right_loc, _ = geod.fwd(*end_loc, end_azimuth - 90, span)

    with rasterio.open(image, "r+") as image_file:
        image_file.crs = "EPSG:4326"
        image_file.transform = from_gcps([
            GroundControlPoint(0, 0, *start_left_loc),
            GroundControlPoint(0, image_file.width, *start_right_loc),
            GroundControlPoint(image_file.height, 0, *end_left_loc),
            GroundControlPoint(image_file.height, image_file.width, *end_right_loc),
        ])
        image_file.descriptions = (desc,)
