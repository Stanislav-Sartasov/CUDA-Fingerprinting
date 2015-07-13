using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDAFingerprinting.Common.BinCylinderCorrelation
{
    public class Cylinder
    {
        public uint[] Values { get; private set; }
        public double Angle { get; private set; }
        public double Norm { get; private set; }

        public Cylinder(uint[] givenValues, double givenAngle, double givenNorm)
        {
            Values = givenValues;
            Angle = givenAngle;
            Norm = givenNorm;
        }
    }

    public class Template
    {
        public Cylinder[] Cylinders { get; private set; }
        public Template(Cylinder[] givenCylinders)
        {
            Cylinders = givenCylinders;
        }
    }

    public class CylinderDatabase
    {
        public Cylinder[] Cylinders { get; private set; }
        public uint[] TemplateIndices { get; private set; }

        public CylinderDatabase(Cylinder[] givenCylinders, uint[] givenTemplateIndices)
        {
            Cylinders = givenCylinders;
            TemplateIndices = givenTemplateIndices;
        }
    }

    public static class BinCylinderCorrelation
    {
        public static double npParamMu = 20;
        public static double npParamTau = 2.0 / 5.0;
        public static int npParamMin = 4, npParamMax = 12;
         
        public static uint bucketsCount = 64;
        public static uint[] buckets = new uint[bucketsCount];
        public static double angleThreshold = Math.PI / 6;

        public static uint[,] bucketMatrix;

        public static int ComputeNumPairs(int template1Count, int template2Count)
        {
            double denom = 1 + Math.Pow(Math.E, -npParamTau * (Math.Min(template1Count, template2Count) - npParamMu));
            return npParamMin + (int)(Math.Round((npParamMax - npParamMin) / denom));
        }

        public static double CalculateCylinderNorm(uint[] cylinder)
        {
            int sum = 0;
            for (int i = 0; i < cylinder.Length; i++)
            {
                if (cylinder[i] == 1)
                {
                    sum++;
                }
            }
            return Math.Sqrt(sum);
        }

        public static uint GetOneBitsCount(uint[] arr)
        {
            uint[] _arr = (uint[])arr.Clone();
            uint count = 0;
            for (int i = 0; i < _arr.Length; i++)
            {
                for (int j = 31; j >= 0; j--)
                {
                    if (_arr[i] % 2 == 1)
                    {
                        count++;
                    }
                    _arr[i] /= 2;
                }
            }
            return count;
        }

        public static double GetAngleDiff(double angle1, double angle2)
        {
            double diff = angle1 - angle2;
            return
                diff < -Math.PI ? diff + 2 * Math.PI :
                diff >= Math.PI ? diff - 2 * Math.PI :
                diff;
        }

        public static double[] GetTemplateCorrelation(Template query, Template[] db)
        {
            double[] similarityRates = new double[db.Length];

            for (int k = 0; k < db.Length; k++) 
            {
                Template templateDb = db[k];

                // Reset buckets array
                // Is this necessary?
                for (int j = 0; j < buckets.Length; j++)
                {
                    buckets[j] = 0;
                }

                foreach (Cylinder queryCylinder in query.Cylinders)
                {
                    foreach (Cylinder cylinderDb in templateDb.Cylinders)
                    {
                        if (GetAngleDiff(queryCylinder.Angle, cylinderDb.Angle) < angleThreshold
                            && queryCylinder.Norm + cylinderDb.Norm != 0)
                        {
                            uint[] givenXOR = queryCylinder.Values.Zip(cylinderDb.Values, (first, second) => first ^ second).ToArray();
                            double givenXORNorm = Math.Sqrt(GetOneBitsCount(givenXOR)); // Bitwise version
                            //double givenXORNorm = CalculateCylinderNorm(givenXOR); // Stupid version

                            uint bucketIndex = (uint)Math.Floor(givenXORNorm / (queryCylinder.Norm + cylinderDb.Norm) * bucketsCount);
                            if (bucketIndex == bucketsCount)
                            {
                                bucketIndex--;
                            }
                            buckets[bucketIndex]++;
                        }
                    }
                }

                int numPairs = ComputeNumPairs(templateDb.Cylinders.Length, query.Cylinders.Length);

                int sum = 0, t = numPairs, i = 0;
                while (i < bucketsCount && t > 0)
                {
                    sum += (int)Math.Min(buckets[i], t) * i;
                    t -= (int)Math.Min(buckets[i], t);
                    i++;
                }
                sum += t * (int)bucketsCount;

                similarityRates[k] = 1 - (float)sum / (numPairs * bucketsCount);
            }

            return similarityRates;
        }

        public static double[] GetTemplateCorrelationOptimized(Template query, CylinderDatabase db, int[] dbTemplateLengths)
        {
            double[] similarityRates = new double[dbTemplateLengths.Length];
            bucketMatrix = new uint[dbTemplateLengths.Length, bucketsCount];

            for (int k = 0; k < db.Cylinders.Length; k++)
            {
                Cylinder cylinderDb = db.Cylinders[k];

                foreach (Cylinder queryCylinder in query.Cylinders)
                {
                    uint[] givenXOR = queryCylinder.Values.Zip(cylinderDb.Values, (first, second) => first ^ second).ToArray();
                    double givenXORNorm = Math.Sqrt(GetOneBitsCount(givenXOR)); // Bitwise version
                    //double givenXORNorm = CalculateCylinderNorm(givenXOR); // Stupid version
                    
                    uint bucketIndex = (uint)Math.Floor(givenXORNorm / (queryCylinder.Norm + cylinderDb.Norm) * bucketsCount);
                    if (bucketIndex == bucketsCount)
                    {
                        bucketIndex--;
                    }

                    bucketMatrix[db.TemplateIndices[k], bucketIndex]++;

                }
            }

            for (int k = 0; k < dbTemplateLengths.Length; k++)
            {
                int numPairs = ComputeNumPairs(dbTemplateLengths[k], query.Cylinders.Length);

                int sum = 0, t = numPairs, i = 0;
                while (i < bucketsCount && t > 0)
                {
                    sum += (int)Math.Min(bucketMatrix[k, i], t) * i;
                    t -= (int)Math.Min(bucketMatrix[k, i], t);
                    i++;
                }
                sum += t * (int)bucketsCount;

                similarityRates[k] = 1 - (float)sum / (numPairs * bucketsCount);
            }

            return similarityRates;
        }
    }
}