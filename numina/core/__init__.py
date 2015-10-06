#
# Copyright 2008-2015 Universidad Complutense de Madrid
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

from .recipes import BaseRecipe
from .recipes import BaseRecipeAutoQC
from .dataframe import DataFrame
from .pipeline import Instrument, Pipeline, InstrumentConfiguration
from .pipeline import ObservingMode
from .objimport import import_object
from .objimport import fully_qualified_name
from .pipelineload import drp_load
from .requirements import DataProductRequirement
from .requirements import Parameter, Requirement
from .products import DataProductType, DataFrameType
from .oresult import ObservationResult
from numina.exceptions import RecipeError # Do not remove, part of the API
from numina.exceptions import ValidationError # Do not remove, part of the API
from .recipeinout import RecipeInput, define_requirements, define_input
from .recipeinout import RecipeResult, define_result
from .recipeinout import ErrorRecipeResult, BaseRecipeResult
from .dataholders import Product
from .oresult import obsres_from_dict
from .qc import QC

# FIXME: these two are deprecated
FrameDataProduct = DataFrameType
DataProduct = DataProductType
