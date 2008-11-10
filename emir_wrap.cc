/* 
 * Copyright 2008 Sergio Pascual
 * 
 * This file is part of PyEMIR
 * 
 * PyEMIR is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * PyEMIR is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with PyEMIR.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

// $Id$


#include <emirdfp/SampleFilter.h>
#include <emirdfp/SampleData.h>


#include <boost/python.hpp>

using namespace boost::python;
using namespace EMIR;

void (SampleFilter::*run1)() const = &SampleFilter::run;
double (SampleFilter::*run2)(int) const = &SampleFilter::run;
SampleData (SampleFilter::*run3)(const SampleData&) const = &SampleFilter::run;
void (SampleFilter::*run4)(const SampleData&, SampleData&) const = &SampleFilter::run;


BOOST_PYTHON_MODULE(emir)
{
    class_<SampleData>("test_data", "This is test_data's docstring")
    .def(init<optional<int> >())
    .add_property("internal", &SampleData::get,  &SampleData::set)
    ;

    class_<SampleFilter>("test_filter", "This is test_filter's docstring")
    .def("run", run1, arg(""), "writes a message")
    .def("run", run2, arg("integer"), "returns input * sqrt(input)")
    .def("run", run3, arg("testdata"), "increases the internal counter of testdata")
    .def("run", run4, (arg("input"),arg("output")), "copies input in outpu and decreases the internal counter") 
    ;
}
