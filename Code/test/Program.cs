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
            for (int i = 1; i <= 110; i++)
            {
                for (int j = 1; j <= 7; j += 4)
                {
                    double[,] imgDoubles = ImageHelper.LoadImage<double>("d://DB2_bmp//DB2_bmp//" + i + "_" + j + ".bmp");
                    double[,] a = CreatTemplateTest(imgDoubles);
                    int[,] bytes = Thinner.Thin(a, a.GetLength(1), a.GetLength(0)).Select2D(x => (int)x);
                    PixelwiseOrientationField field = new PixelwiseOrientationField(bytes, 16);
                    List<Minutia> minutias = MinutiaDetector.GetMinutias(bytes, field);
                    TemplateCreator creator = new TemplateCreator(minutias);
                    Template[] t = { creator.CreateTemplate() };
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
            }
            Console.WriteLine("min count fot 1 - {0} ({1})", min, countmin);
            Console.WriteLine("max count for < 1 - {0} ({1})", max, countmax);
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
    }
}
