/* 
 * Copyright 2008 Sergio Pascual
 * 
 * This file is part of PyEmir
 * 
 * PyEmir is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * PyEmir is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with PyEmir.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

// $Id$


#include <boost/python.hpp>

using namespace boost::python;
namespace py = boost::python;

int test(numeric::array& nsP)
{
  object shape = nsP.getshape();
  int rows = extract<int>(shape[0]);
  int cols = extract<int>(shape[1]);
  return cols;
}

void exercise(numeric::array& y)
{
  y[py::make_tuple(2,1)] = 3;
}

numeric::array new_array()
{
  return numeric::array(py::make_tuple(py::make_tuple(1, 2, 3) ,
      py::make_tuple(4, 5, 6) , py::make_tuple(7, 8, 9) ) );
}

void info(numeric::array const& z)
{
  z.info();
}

BOOST_PYTHON_MODULE(test)
{
  using namespace boost::python;

  numeric::array::set_module_and_type("numpy", "ndarray");

  def("test", &test);
  def("exercise", &exercise);
  def("new_array", &new_array);
  def("info", &info);
}
