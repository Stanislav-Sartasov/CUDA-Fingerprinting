using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDAFingerprinting.Common.BinCylinderCorrelation
{
    public static class BinTemplateCorrelation
    {
        public static int npParamMin = 11, npParamMax = 13;
        public static double npParamMu = 30;
        public static double npParamTau = 2.0 / 5.0;
         
        public static uint bucketsCount = 64;
        public static uint[] buckets = new uint[bucketsCount];
        public static double angleThreshold = Math.PI / 6;

        public static uint[,] bucketMatrix;

        public static int ComputeNumPairs(int template1Count, int template2Count)
        {
            double denom = 1 + Math.Pow(Math.E, -npParamTau * (Math.Min(template1Count, template2Count) - npParamMu));
            return npParamMin + (int)(Math.Round((npParamMax - npParamMin) / denom));
        }

        public static double[] GetTemplateCorrelationMultiple(Template query, Template[] db)
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
                        if (CylinderHelper.GetAngleDiff(queryCylinder.Angle, cylinderDb.Angle) < angleThreshold
                            && queryCylinder.Norm + cylinderDb.Norm != 0)
                        {
                            uint[] givenXOR = queryCylinder.Values.Zip(cylinderDb.Values, (first, second) => first ^ second).ToArray();
                            double givenXORNorm = Math.Sqrt(CylinderHelper.GetOneBitsCount(givenXOR)); // Bitwise version
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

        public static double[] GetTemplateCorrelationMultipleOptimized(Template query, CylinderDatabase db, int[] dbTemplateLengths)
        {
            double[] similarityRates = new double[dbTemplateLengths.Length];
            bucketMatrix = new uint[dbTemplateLengths.Length, bucketsCount];

            for (int k = 0; k < db.Cylinders.Length; k++)
            {
                Cylinder cylinderDb = db.Cylinders[k];

                foreach (Cylinder queryCylinder in query.Cylinders)
                {
                    uint[] givenXOR = queryCylinder.Values.Zip(cylinderDb.Values, (first, second) => first ^ second).ToArray();
                    double givenXORNorm = Math.Sqrt(CylinderHelper.GetOneBitsCount(givenXOR)); // Bitwise version
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