using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TriangulationWithAfineTransformation.Classes
{
    internal class TriangulationsMatcher
    {
        private List<Triangle> TrianglesFrom;
        private List<Triangle> TrianglesTo;

        public TriangulationsMatcher(List<Triangle> trianglesFrom, List<Triangle> trianglesTo)
        {
            TrianglesFrom = trianglesFrom;
            TrianglesTo = trianglesTo;
        }

        private OptimumConversionBuilder CreateOptimumBuilder(Triangle from, List<Triangle> list) 
        {
            OptimumConversionBuilder ocb = null;

            foreach (Triangle t in list)
            {
                OptimumConversionBuilder ocbTmp = new OptimumConversionBuilder(from, t);
                if (ocb == null || ocbTmp.Distance < ocb.Distance)
                    ocb = ocbTmp;
            }

            return ocb;
        }

        public double[] Match(double distance, int threshhold) 
        {
            double[] result = null;

            foreach (Triangle tFrom in TrianglesFrom)
            {
                double[] tmpResult = new double[3];

                List<Triangle> copyFrom = new List<Triangle>(TrianglesFrom);
                List<Triangle> copyTo = new List<Triangle>(TrianglesTo);

                OptimumConversionBuilder ocb = CreateOptimumBuilder(tFrom, copyTo);
                
                if (ocb.Distance > distance)
                    continue;

                List<Triangle> UpdatedTriangles = new List<Triangle>();
                foreach (Triangle t in copyFrom)
                    UpdatedTriangles.Add(tFrom.GetTransformation(ocb.Dx, ocb.Dy, ocb.Phi));

                foreach (Triangle t in UpdatedTriangles) 
                {
                    Triangle equals = FindEquals(t, copyTo);
                    if (equals != null)
                        tmpResult[0]++;
                    else
                    {
                        Triangle near = FindNear(t, distance, copyTo);
                        if (near != null)
                            tmpResult[1]++;
                    }
                }

                if (result == null)
                {
                    result = new double[3];
                    result[0] = tmpResult[0];
                    result[1] = tmpResult[1];
                }

                if (result[0] + result[1] < tmpResult[0] + tmpResult[1])
                {
                    result[0] = tmpResult[0];
                    result[1] = tmpResult[1];
                }

                if (100* (result[0] + result[1])/TrianglesFrom.Count > threshhold)
                    break;
            }

            return result;
        }

        private Triangle FindEquals(Triangle triangle, List<Triangle> list) {
            Triangle result = null;

            foreach (Triangle t in list) 
            {
                if (t.Equals(triangle))
                    return t;
            }

            return result;
        }

        private Triangle FindNear(Triangle triangle, double distance, List<Triangle> list) {
            Triangle result = null;
            double nearDistance = -1;
            foreach (Triangle t in list)
            {
                if (t.Equals(triangle, distance))
                { 
                    double tmpNearDistance = t.GetDistanceTo(triangle);
                    if (tmpNearDistance < nearDistance || nearDistance == -1)
                    {
                        result = t;
                        nearDistance = tmpNearDistance;
                    }
                }
            }
            return result;
        }
    }
}