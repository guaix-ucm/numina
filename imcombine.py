#!/usr/bin/env python

#
# Copyright 2008 Sergio Pascual
# 
# This file is part of PyEmir
# 
# PyMilia is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# PyEmir is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with PyEmir.  If not, see <http://www.gnu.org/licenses/>.
# 

# $Id$

'''Imcombine docstring'''

from __future__ import with_statement
import getopt, sys
import logging
import logging.config

import pyfits
import os

import scipy
import scipy.stsci
import scipy.stsci.image as img

import mydirs

logging.config.fileConfig(os.path.join(mydirs.confdir, "logging.conf"))
lg = logging.getLogger("imcombine")

def subtract_fits(a, b, result):
    '''Subtract fits'''
    # get header and data of the first image
    aa, ha = pyfits.getdata(a, header=True)
    c = aa - pyfits.getdata(b)
    # save the header and the data in result
    lg.debug('Writing to disk %s' % result)
    try:
        pyfits.writeto(result, data=c, header=ha, output_verify='ignore', clobber=True)
        lg.debug('Done %s', result)
    except IOError, strrerror:
        lg.error(strrerror)

def compute_flat(flatlist):
    flatdata = [pyfits.getdata(i) for i in flatlist]    
    flatdata = [i / i.mean() for i in flatdata]
    result = img.median(flatdata)
    return result

def divide_by_flat_data(a, b, result):
    '''Divide a fits image by a numpy array'''
    # get header and data of the first image
    aa, ha = pyfits.getdata(a, header=True)
    c = aa / b
    # save the header and the data in result
    lg.debug('Writing to disk %s' % result)
    try:
        pyfits.writeto(result, data=c, header=ha, output_verify='ignore', clobber=True)
        lg.debug('Done %s', result)
    except IOError, strrerror:
        lg.error(strrerror)
    pass


def update_name(name, flag):
    return name + '.' + flag


def main():
    '''This is the docstring'''
    
    # Process command line
    inputfile = None
    try:
        inputfile = sys.argv[1]
    except IndexError, err:
        lg.error('We need one file in the command line')
        sys.exit(2)
    
    lg.info('Input file %s' % inputfile)
    
    # Read input file
    procfiles = []    
    try:
        with open(inputfile) as f:
            for line in f:
                if not line.startswith('#'):
                    e = line.split() 
                    (use,name,dark,x,y) = (bool(e[0]),e[1],e[2],int(e[3]),int(e[4]))
                    lg.debug('Processing %s' % name)
                    procfiles.append((use,name,dark,x,y))
    except IOError, strrerror:
        lg.error(strrerror)
          
    lg.info('%d files to be processed' % len(procfiles))
    
    # Dark subtraction
    lg.info('Subtracting dark')
    for i in procfiles:
        if not i[0]:
            lg.debug('Skipping %s' % (i[1]))
        else:
            lg.debug('Processing %s with %s' % (i[1],i[2]))
            subtract_fits(i[1], i[2], update_name(i[1], 'D'))
            

    lg.info('Computing flat')
    flatlist = []
    for i in procfiles:
        if not i[0]:
            lg.debug('Skipping %s' % (i[1]))
        else:
            flatlist.append(update_name(i[1], 'D'))
            
    flatdata = compute_flat(flatlist)
    lg.debug('Writing flat to disk as %s' % 'Flat.fits.0')
    try:
        pyfits.writeto('Flat.fits.0', data=flatdata, clobber=True)
        lg.debug('Done %s', 'Flat.fits.0')
    except IOError, strrerror:
        lg.error(strrerror)
    
    lg.info('Dividing by flat')
    lg.info('Flat statistics %f' % flatdata.mean())
    for i in procfiles:
        if not i[0]:
            lg.debug('Skipping %s' % (i[1]))
        else:
            lg.debug('Processing %s with %s' % (update_name(i[1], 'D'), 'flat'))            
            divide_by_flat_data(update_name(i[1], 'D'), flatdata, update_name(i[1], 'DF'))
        
    

if __name__ == "__main__":
    main()


