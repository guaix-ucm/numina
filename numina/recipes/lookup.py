#
# Copyright 2008-2012 Universidad Complutense de Madrid
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

'''
Generic functions used by numina CLI to look up values of the requirements
'''

from .requirements import Requirement, Parameter, DataProductParameter

from numina.generic import generic

@generic
def lookup(req, source):
    '''Look up values of the requirements.'''
    
    # We don't have a default implementation
    raise NotImplementedError

@lookup.register(Requirement)
def _lookup_req(req, source):    
    if req.name in source:
        # FIXME: add validation
        return source[req.name]
    elif req.optional:
        return None
    elif req.value is not None:
        return req.value
    else:
        raise LookupError('parameter %s must be defined' % req.name)

@lookup.register(Parameter)
def _lookup_param(req, source):    
    if req.name in source:
        # FIXME: add validation
        return source[req.name]
    elif req.optional:
        return None
    else:
        return req.value
        
        
@lookup.register(DataProductParameter)
def _lookup(req, source):
    if req.name in source:
        # FIXME: add validation
        return source[req.name]
    elif req.optional:
        return None
    elif req.default is not None:
        return req.default
    else:
        raise LookupError('parameter %s must be defined' % req.name)
