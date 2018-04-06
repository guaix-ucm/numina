# Sutherland–Hodgman algorithm implementation in Cython
# https://stackoverflow.com/questions/44765229/working-with-variable-sized-lists-with-cython

cdef compute_intersection(double cp10, double cp11, double cp20, double cp21,
                          double s0, double s1, double e0, double e1):
  dc0, dc1 = cp10 - cp20, cp11 - cp21
  dp0, dp1 =  s0 - e0, s1 - e1
  n1 = cp10 * cp21 - cp11 * cp20
  n2 = s0 * e1 - s1 * e0
  n3 = 1.0 / (dc0 * dp1 - dc1 * dp0)
  return (n1*dp0 - n2*dc0) * n3, (n1*dp1 - n2*dc1) * n3

cdef cclipix(subjectPolygon, clipPolygon):
   cdef double cp10, cp11, cp20, cp21
   cdef double s0, s1, e0, e1
   cdef double s_in, e_in

   outputList = subjectPolygon
   cp10, cp11 = clipPolygon[-1]

   for cp20, cp21 in clipPolygon:

      inputList = outputList
      outputList = []
      s0, s1 = inputList[-1]
      s_in = (cp20 - cp10) * (s1 - cp11) - (cp21 - cp11) * (s0 - cp10)
      for e0, e1  in inputList:
         e_in = (cp20 - cp10) * (e1 - cp11) - (cp21 - cp11) * (e0 - cp10)
         if e_in > 0:
            if s_in <= 0:
               outputList.append(
                   compute_intersection(cp10, cp11, cp20, cp21, s0, s1, e0, e1)
               )
            outputList.append((e0, e1))
         elif s_in > 0:
            outputList.append(
                compute_intersection(cp10, cp11, cp20, cp21, s0, s1, e0, e1)
            )
         s0, s1, s_in = e0, e1, e_in
      if len(outputList) < 1:
          return []
      cp10, cp11 = cp20, cp21

   return outputList


def _clippix(subjectPolygon, clipPolygon):
    return cclipix(subjectPolygon, clipPolygon)


def _polygon_area(polygon):
    """Signed area of a planar non-self-intersecting polygon.
    
    See http://mathworld.wolfram.com/PolygonArea.html for details.
    """
    cdef double x1, y1, x2, y2, area
    
    if len(polygon) == 0:
        return 0.0

    area = 0.0
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        area += x1 * y2 - x2 * y1
        x1, y1 = x2, y2
        
    return area/2
