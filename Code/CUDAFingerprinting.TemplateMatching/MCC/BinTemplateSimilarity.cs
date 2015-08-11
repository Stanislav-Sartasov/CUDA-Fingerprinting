using System;
using System.Linq;

namespace CUDAFingerprinting.TemplateMatching.MCC
{
    public class BinTemplateSimilarity
    {
        public static void PrintMatrix(uint[,] arr)
        {
            int rowLength = arr.GetLength(0);
            int colLength = arr.GetLength(1);

            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < colLength; j++)
                {
                    Console.Write(string.Format("{0} ", arr[i, j]));
                }
                Console.Write(Environment.NewLine + Environment.NewLine);
            }
        }

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

        public static double[] GetTemplateSimilarity(Template query, Template[] db)
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

        public static double[] GetTemplateSimilarityWithMask(Template query, Template[] db)
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

                for (int countI = 0; countI < query.Cylinders.Length; countI += 2)
                {
                    for (int countJ = 0; countJ < templateDb.Cylinders.Length; countJ += 2)
                    {
                        if (CylinderHelper.GetAngleDiff(query.Cylinders[countI].Angle, templateDb.Cylinders[countJ].Angle) <= angleThreshold)
                        {
                            uint[] common =
                                query.Cylinders[countI + 1].Values.Zip(templateDb.Cylinders[countJ + 1].Values,
                                    (first, second) => first & second).ToArray();
                            uint[] firstAndSecond = query.Cylinders[countI].Values.Zip(common,
                                    (first, second) => first & second).ToArray();
                            uint[] secondAndFirst = templateDb.Cylinders[countJ].Values.Zip(common,
                                    (first, second) => first & second).ToArray();
                            double givenFristNorm = Math.Sqrt(CylinderHelper.GetOneBitsCount(firstAndSecond));
                            double givenSecondNorm = Math.Sqrt(CylinderHelper.GetOneBitsCount(secondAndFirst));
                            
                            uint[] givenXOR = firstAndSecond.Zip(secondAndFirst, (first, second) => first ^ second).ToArray();

                            double givenXORNorm = Math.Sqrt(CylinderHelper.GetOneBitsCount(givenXOR)); // Bitwise version
                            //double givenXORNorm = CalculateCylinderNorm(givenXOR); // Stupid version


                            uint bucketIndex = (uint)Math.Floor(givenXORNorm / (givenFristNorm + givenSecondNorm) * bucketsCount);
                            if (bucketIndex == bucketsCount)
                            {
                                bucketIndex--;
                            }
                            buckets[bucketIndex]++;
                            
                        }
                    }
                }

                int numPairs = ComputeNumPairs(templateDb.Cylinders.Length/2, query.Cylinders.Length/2);

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

        public static double[] GetTemplateSimilarityOptimized(Template query, CylinderDatabase db, int[] dbTemplateLengths)
        {
            double[] similarityRates = new double[dbTemplateLengths.Length];
            bucketMatrix = new uint[dbTemplateLengths.Length, bucketsCount];

            for (int k = 0; k < db.Cylinders.Length; k++)
            {
                Cylinder cylinderDb = db.Cylinders[k];

                foreach (Cylinder queryCylinder in query.Cylinders)
                {

                    uint[] givenXOR = queryCylinder.Values.Zip(cylinderDb.Values, (first, second) => first ^ second).ToArray();

                    //for (int i = 0; i < givenXOR.Length; i++)
                    //{
                    //    Console.Write(givenXOR[i] + ", ");
                    //}
                    //Console.WriteLine();
                    uint oneBitsCount = CylinderHelper.GetOneBitsCount(givenXOR);
                    //Console.Write(oneBitsCount + " ");

                    double givenXORNorm = Math.Sqrt(oneBitsCount); // Bitwise version
                    //double givenXORNorm = CalculateCylinderNorm(givenXOR); // Stupid version

                    if (CylinderHelper.GetAngleDiff(queryCylinder.Angle, cylinderDb.Angle) < angleThreshold
                        && queryCylinder.Norm + cylinderDb.Norm != 0)
                    {
                        uint bucketIndex = (uint)Math.Floor(givenXORNorm / (queryCylinder.Norm + cylinderDb.Norm) * bucketsCount);
                        //if (bucketIndex >= 63)
                        //{
                        //    Console.Write("LOOOOL");
                        //    Console.WriteLine(k);
                        //}
                        if (bucketIndex == bucketsCount)
                        {
                            bucketIndex--;
                        }

                        uint row = db.TemplateIndices[k];
                        bucketMatrix[row, bucketIndex]++;
                    }
                }
            }

            //Console.WriteLine("END");

            PrintMatrix(bucketMatrix);

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