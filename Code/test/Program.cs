using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.OrientationField;
using CUDAFingerprinting.FeatureExtraction.Minutiae;
using CUDAFingerprinting.ImageProcessing.GaborEnhancement;
using CUDAFingerprinting.FeatureExtraction.Tests;
using CUDAFingerprinting.ImageProcessing.Binarization;
using CUDAFingerprinting.ImageProcessing.ImageThinning;
using CUDAFingerprinting.FeatureExtraction.TemplateCreate;
using CUDAFingerprinting.TemplateMatching;
using CUDAFingerprinting.TemplateMatching.MCC;

namespace test
{
    class Program
    {
        static void Main(string[] args)
        {
            int countmax = 0;
            int countmin = 0;
            int max = 0;
            int min = 100;
            /*for (int i = 1; i <= 110; i++)
            {
                for (int j = 1; j <= 7; j += 4)
                {
                    double[,] imgDoubles = ImageHelper.LoadImage<double>("d://DB2_bmp//" + i + "_" + j + ".bmp");
                    double[,] a = CreatTemplateTest(imgDoubles);
                    int[,] bytes = Thinner.Thin(a, a.GetLength(1), a.GetLength(0)).Select2D(x => (int)x);
                    PixelwiseOrientationField field = new PixelwiseOrientationField(bytes, 16);
                    List<Minutia> minutias = MinutiaDetector.GetMinutias(bytes, field);
                    foreach (var minutia in minutias)
                    {
                        Console.WriteLine(minutia.X + ", " + minutia.Y + ", " + minutia.Angle + ", ");
                    }
                    Console.WriteLine();
                    TemplateCreator creator = new TemplateCreator(minutias);
                    Template[] t = { creator.CreateTemplate() };
                    foreach (var template in t)
                    {
                        for (int k = 0; k < template.Cylinders.Length; k+=2)
                        {
                            PrintCylinder(template.Cylinders[k].Values, template.Cylinders[k + 1].Values);
                        }
                    }
                    if (BinTemplateSimilarity.GetTemplateSimilarity(t[0], t)[0].Equals(1))
                    {
                        min = min > t[0].Cylinders.Count() ? t[0].Cylinders.Count() : min;
                        countmin++;
                    }
                    else
                    {
                        countmax++;
                        max = max < t[0].Cylinders.Count() ? t[0].Cylinders.Count() : max;
                    }
                }
            }*/

            Minutia minutia = new Minutia();
            List<Minutia> minutiae = new List<Minutia>();
            for (int i = 0; i < 100; i++)
            {
                minutia.X = i+1;
                minutia.Y =(int)Math.Sin((i+1));
                minutia.Angle = (float)(i*0.3);
                minutiae.Add(minutia);
            }
            TemplateCreator creator = new TemplateCreator(minutiae);
            Template[] t = { creator.CreateTemplate() };
            Console.WriteLine(t[0].Cylinders.Length);
            Console.ReadKey();
        /*    int count = 1;
            foreach (var template in t)
            {
                for (int k = 0; k < template.Cylinders.Length; k += 2)
                {
                    PrintCylinder(template.Cylinders[k].Values, template.Cylinders[k + 1].Values, count);
                    count++;
                }
            }*/

       /*     Console.WriteLine("min count fot 1 - {0} ({1})", min, countmin);
            Console.WriteLine("max count for < 1 - {0} ({1})", max, countmax);*/
            // ImageHelper.SaveArrayToBitmap(CreatTemplateTest(a)).Save("d://hglf23.bmp");


            /* imgDoubles = ImageHelper.LoadImage<double>("d://DB2_bmp//DB2_bmp//3_4.bmp");
             a = CreatTemplateTest(imgDoubles);
             bytes = Thinner.Thin(a, a.GetLength(1), a.GetLength(0)).Select2D(x => (int)x);
             field = new PixelwiseOrientationField(bytes, 16);
            
             List<Minutia> minutiae2 = MinutiaDetector.GetMinutias(bytes, field);
             TemplateCreator creator2 = new TemplateCreator(minutiae2);*/
            Console.ReadKey();
            // seg.SaveSegmentation(seg.MakeBitmap(seg.Segmentate()),"d://123.bmp");
        }

        public static double[,] CreatTemplateTest(double[,] imgDoubles)
        {
            imgDoubles.DoNormalization(100, 100);

            int[,] imgInts = imgDoubles.Select2D((x => (int)x));
            OrientationField orf = new OrientationField(imgInts, 16);
            double[,] orient = orf.GetOrientationMatrix(imgInts.GetLength(0), imgInts.GetLength(1));

            var freqMatrx = LocalRidgeFrequency.GetFrequencies(imgDoubles, orient);

            var res = ImageEnhancement.Enhance(imgDoubles, orient, freqMatrx, 32, 8);

            return ImageBinarization.Binarize2D(res, 128);
        }

        public static void PrintCylinder(uint[] cylinderValue, uint[] cylinderMask, int number)
        {
            Cylinder3D cylinder3DValue = new Cylinder3D();
            Cylinder3D cylinder3DMask = new Cylinder3D();
            cylinder3DValue.Cylinder = cylinderValue;
            cylinder3DMask.Cylinder = cylinderMask;
            FileStream file = new FileStream("cylinder" + number + ".txt" , FileMode.Create, FileAccess.ReadWrite);
            StreamWriter write = new StreamWriter(file);
            for (int k = 1; k <= TemplateCreator.HeightCuboid; k++)
            {
                for (int i = 1; i <= TemplateCreator.BaseCuboid; i++)
                {
                    for (int j = 1; j <= TemplateCreator.BaseCuboid; j++)
                    {
                        write.Write(cylinder3DValue.GetValue(i, j, k) + cylinder3DMask.GetValue(i, j, k));
                    }
                    write.WriteLine();
                }
                write.WriteLine();
            }
            write.Close();
        }
    }
}
