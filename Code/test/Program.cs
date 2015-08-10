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
            double[,] imgDoubles = ImageHelper.LoadImage<double>("d://DB2_bmp//62_1.bmp");
            double[,] a = CreatTemplateTest(imgDoubles);
           // ImageHelper.SaveArrayToBitmap(CreatTemplateTest(a)).Save("d://hglf23.bmp");
            
           int[,] bytes = Thinner.Thin(a, a.GetLength(1), a.GetLength(0)).Select2D(x => (int)x);
            PixelwiseOrientationField field = new PixelwiseOrientationField(bytes, 16);
            List<Minutia> minutias = MinutiaDetector.GetMinutias(bytes, field);
            TemplateCreator creator = new TemplateCreator(minutias);

            imgDoubles = ImageHelper.LoadImage<double>("d://DB2_bmp//62_3.bmp");
            a = CreatTemplateTest(imgDoubles);
            bytes = Thinner.Thin(a, a.GetLength(1), a.GetLength(0)).Select2D(x => (int)x);
            field = new PixelwiseOrientationField(bytes, 16);
            
            List<Minutia> minutiae2 = MinutiaDetector.GetMinutias(bytes, field);
            TemplateCreator creator2 = new TemplateCreator(minutiae2);
            Template[] t = {creator.CreateTemplate()};
            Console.WriteLine(BinTemplateSimilarity.GetTemplateSimilarityWithMask(creator2.CreateTemplate(), t)[0]);
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
