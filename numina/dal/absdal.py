#
# Copyright 2014 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

'''DAL base class'''

from ..core.oresult import ObservationResult
from .daliface import DALInterface


class AbsDAL(DALInterface):

    def obsres_from_proc_oblock_id(self, instrument, child_id):
        # This implementation does not depend
        # on the details of the DAL...
        ob = self.search_oblock_from_id(child_id)
        prod = self.search_prod_obsid(instrument, child_id, 'default')

        h = ObservationResult(ob.mode)
        h.id = child_id
        h.instrument = ob.instrument
        h.parent = ob.parent
        h.update_with_product(prod)
        return h
