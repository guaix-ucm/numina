# #
# # Copyright 2008-2015 Universidad Complutense de Madrid
# #
# # This file is part of Numina
# #
# # Numina is free software: you can redistribute it and/or modify
# # it under the terms of the GNU General Public License as published by
# # the Free Software Foundation, either version 3 of the License, or
# # (at your option) any later version.
# #
# # Numina is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# # GNU General Public License for more details.
# #
# # You should have received a copy of the GNU General Public License
# # along with Numina.  If not, see <http://www.gnu.org/licenses/>.
# #
#
# """Two basic functions to store frames and arrays"""
#
#
# import warnings
#
# import numpy
#
#
# def dump_dataframe(obj, where):
#     # save fits file
#     if obj.frame is None:
#         # assume filename contains a FITS file
#         return None
#     else:
#         if obj.filename:
#             filename = obj.filename
#         elif 'FILENAME' in obj.frame[0].header:
#             filename = obj.frame[0].header['FILENAME']
#         elif hasattr(where, 'destination'):
#             filename = where.destination + '.fits'
#         else:
#             filename = where.get_next_basename('.fits')
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')
#             obj.frame.writeto(filename, clobber=True)
#         return filename
#
#
# def dump_numpy_array(obj, where):
#     # FIXME:
#     #filename = where.get_next_basename('.txt')
#     filename = where.destination + '.txt'
#     numpy.savetxt(filename, obj)
#     return filename
#
#
# # FIXME: remove this function together with init_dump_backends
# def load_cli_storage():
#     pass