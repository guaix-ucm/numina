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

from numina.core import ObservationResult
from .daliface import DALInterface
from .tagger import tagger


class AbsDAL(DALInterface):

    def obsres_from_oblock_id(self, obsid):
        # This implementation does not depend
        # on the details of the DAL...
        ob = self.search_oblock_from_id(obsid)
        ch = [self.obsres_from_proc_oblock_id(ob.instrument, cid)
              for cid in ob.children]

        h = ObservationResult(obsid)
        h.instrument = ob.instrument
        h.mode = ob.mode
        h.parent = ob.parent
        h.tags = {}
        h.files = ob.files
        h.children = ch

        tags_for_this_mode = tagger(ob.instrument, ob.mode)

        master_tags = tags_for_this_mode(h)
        h.tags = master_tags
        return h

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
