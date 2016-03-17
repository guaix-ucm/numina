Version 0.14 (4 Feb 2016)
=========================

Changes
-------
* Remove 'worker' module
* Rework 'store' module
* Changed the format of 'control.yaml'
* Removed dependency on 'singledispatch'
* Added wavecalibration module
* RecipeRequirement is now RecipeInput (GCS compatibility)
* RecipeInputBuilder is a method of Recipe (GCS compatibility)
* MIgrated to astropy 1.0


Version 0.13.2 (22 Apr 2015)
----------------------------
 Changes:
  * Support only astropy 0.4.x
  * Add Solaris 10 documentation

Version 0.13.1 (27 Jan 2015)
----------------------------
 Fixes:
  * Portability, code compiles with clang (#113)

Version 0.13.0 (26 Jan 2015)
----------------------------
 Changes:
  * Initial DAL implementation
  * Ported to use py.test
  * Ported to astropy 0.4
  * New method to create automatically Requirements
    and Products
  * New recentering routine
  * New routine for FHWM computation

Version 0.12.2 (17 Oct 2014)
----------------------------
 Fixes:
  * PEP8 fixes in the code

Version 0.12.1 (30 Apr 2014)
----------------------------
 Fixes:
  * Include Cython-generated code in distribution

Version 0.12.0 (30 Apr 2014)
----------------------------
 Changes:
  * Initial implementation of FITS Schema
  * Allow plain python types in Product
  * Import ArrayType from PyEmir
  * Import recenter function from PyEmir
  * Handle multiextension FITS in promote_hdulist
 Fixes:
  * Data was not copied back in processing nodes

Version 0.11.0 (02 Apr 2014)
----------------------------
Changes:
 * Ported to astropy (0.3)
 * Added new filters (DivideByExposure, SkyCorrector)
 * Provide custom extension to data types (required to interoperate with GTC)
 * New EnumType that mimicks Enum from Python 3.4
 * numina.QA is now numina.core.QC and it's an Enum
 * Testing recipes AlwaysFailRecipe and AlwaysSuccessRecipe added
 * Extended Requirements (ObservationResultRequirement)
 * Recipes take only one input (of type RecipeRequirement)
 * RecipeResult and RecipeRequirement has a dictionary interface
 * Attributes __requires__  and __provides__ of Recipes removed

Version 0.10.3 (16 Jan 2014)
----------------------------
Changes:
 * Fix typo in reciperesult.py

Version 0.10.2 (08 Nov 2013)
----------------------------
Changes:
 * RecipeRequires and RecipeResult use metaclasses
   instead of hacking __new__

Version 0.10.1 (29 Oct 2013)
----------------------------
Bugfix:
 * Handle gracefully if Cython is missing
 * Cython in Mac does not get numpy_includes

Version 0.10.0 (28 Oct 2013)
----------------------------
Changes:
 * nIR detector modes (Ramp, Fowler, etc.)
 * Instrument load improved

Version 0.9.1 (05 Dec 2012)
---------------------------
Bugfix:
 * Error importing ConfigParserError

Version 0.9.0 (05 Dec 2012)
Changes:
 * Rewritten plugin load system
 * Instrument, Pipeline and Recipe classes changed
 * New CLI interface
    - No need to load instruments.yaml
    - Files are copied from datadir to workdir

Version 0.8.7 (19 Nov 2012)
Bugfixes:
 * FrameInfo can be read from a list or string

Version 0.8.6 (13 Nov 2012)
Bugfixes:
  * Handle 32bits systems without float128

Version 0.8.5 (13 Nov 2012)
Changes:
  * Removed download_url from setup.py
  * Fixed bad handling of badpixels in nIR readout preprocessing routines

Version 0.8.4 (07 Nov 2012)
Changes:
  * Updated tests for ramp_array and fowler_array
  * C module creation compatible with Python3

Version 0.8.3 (24 Sep 2012)
Bugfixes:
  * #109, 110: bugs that prevented installing in Mac OS X

Version 0.8.2 (17 Sep 2012)
Changes:
  * New function 'process_ramp' to process images in follow-up-the-ramp mode
  * Updated to use PyCapsule instead of PyCObject
  * Use PyMem_Malloc aand PyMem_Free where appropriated

Bugfixes:
  * Removed wrong term in weighted sample variance

Version 0.8.1 (12 Jul 2012)

Changes:
  * New functions 'cosmetics' and 'ccdmask' to find bad pixels
  * Pyfits warning about overwritting files hidden


Version 0.8.0 (18 Jun 2012)

Changes:
 
  * new format of the recipe input and recipe results
  * new command 'show' in CLI


Version 0.7.0 (20 May 2012)
---------------------------

Changes:

  * using namespace package numina.pipelines to hold pipelines


Version 0.6.1 (15 May 2012)
---------------------------

Changes:

  * lookup is a generic function
  * added tests for generic
  * fixed a bug in default implementation of generic

Version 0.6.0 (11 May 2012)
---------------------------

Changes:

  * Removed legacy code
  * YAML default serialization format
  * Changes recipe API
  * Added pipelines
  * Supports GCC 4.7


Version 0.5.0 (27 Oct 2011)
---------------------------

Changes:
 * Pyemir split from Numina
 * Bug fixes to work with Pyemir and Pontifex

Version 0.4.2 (07 Oct 2011)
---------------------------

Changes:
 * Fixed error with object mask creation
 * Added numdisplay to required packages

Version 0.4.1 (23 Sep 2011)
---------------------------

Changes:
 * Allows installation using pip

Version 0.4.0 (7 Sep 2011)
--------------------------

Changes:
 * Direct image implemented
 * Minor bugs and fixes
   
Version 0.3.0 (24 Feb 2011)
---------------------------

Changes:
 * Implemented some recipes for detector characterization
 * Full treatment of EMIR detector amplifiers
 * Module names follow PEP8
 * Surface fitting routines
 * Working methods in combine:
   - Median
   - Average
   - Minamax
   - Sigclip
 
Version 0.2.5 (09 Sep 2010)
---------------------------

Changes:
 * Combine internals changed
 * New method to load recipes, based in subclasses
 * Recipe classes announce their capabilities

Version 0.2.4 (08 Jul 2010)
---------------------------

Changes:
 * Parameter-passing API for Recipes has been changed.
 * JSON serialization format has been changed.
 * New functions to request parameters and schema information 
   (numina.recipes.registry and numina.recipes.schema)
 * Parallel version of map (para_map) in numina.worker   
 
Version 0.2.3 (13 Apr 2010) Bugfix release
------------------------------------------

Bugfixes:
 * #94  Missing header files inside src
 * Errors in documentation fixed

Version 0.2.2 (13 Apr 2010) Bugfix release
------------------------------------------

Bugfixes:
 * #91  Error creating object mask in direct_imaging
 * Doctest errors fixed

Enhancements: 
 * #86 Combines images using extinction
 * store function uses custom generic function (is extensible)
 * repository migrated to mercurial

Version 0.2.1
-------------

(15 March 2010, from /pyemir/trunk revision 647)
https://guaix.fis.ucm.es/svn-private/emir/pyemir/tags/0.2.1/

Bugfixes: #89, pkgutil.get_data not present in python 2.5 


Version 0.2.0
-------------

(18 February 2010, from /pyemir/trunk revision 639)
https://guaix.fis.ucm.es/svn-private/emir/pyemir/tags/0.2.0/

direct_image implemented
Multidimensional GuassianProfile with tests
Simulation tools moved to numina


Version 0.1.0
-------------

(08 February 2010, from /pyemir/trunk revision 627)
https://guaix.fis.ucm.es/svn-private/emir/pyemir/tags/0.1.0/

Internal release, it includes the documentation of the Recipes and a bare bones recipe runner
The performance of _combine has been increased in a factor of 2 


Version 0.0.6
-------------
(27 January 2010, from /pyemir/trunk revision 602)
https://guaix.fis.ucm.es/svn-private/emir/pyemir/tags/0.0.6/

Internal release

Version 0.0.5
-------------

(27 January 2010, from /pyemir/trunk revision 596)
https://guaix.fis.ucm.es/svn-private/emir/pyemir/tags/0.0.5/

Bugfixes: #53, false result in direct_image

Version 0.0.4
-------------
(27 January 2010, from /pyemir/trunk revision 595)
https://guaix.fis.ucm.es/svn-private/emir/pyemir/tags/0.0.4/

Internal release

Version 0.0.3
-------------
(26 January 2010, from /pyemir/trunk revision 586)
https://guaix.fis.ucm.es/svn-private/emir/pyemir/tags/0.0.3/

Internal release

Version 0.0.2
-------------
(12 November 2009, from /pyemir/trunk revision 516)
https://guaix.fis.ucm.es/svn-private/emir/pyemir/tags/0.0.2/

Internal release

Version 0.0.1
-------------
(12 March 2009, from /pyemir/trunk revision 413)
https://guaix.fis.ucm.es/svn-private/emir/pyemir/tags/0.0.1/

Internal release
