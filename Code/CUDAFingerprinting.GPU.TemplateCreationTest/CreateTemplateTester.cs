using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using CUDAFingerprinting.Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.FeatureExtraction;
using CUDAFingerprinting.FeatureExtraction.TemplateCreate;
using CUDAFingerprinting.TemplateMatching.MCC;

namespace CreateTemplateTest
{
    [TestClass]
    public class CreateTemplateTester
    {
        [DllImport("CUDAFingerprinting.GPU.TemplateCreation.dll", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "createTemplate")]
        public static extern void createTemplate(Minutia[] minutiae, int minutiaeLenght, out Cylinder[] cylinders,
            out int cylinderLenght);

        [TestMethod]
        public void CreateTemplateTest()
        {
            Minutia[] minutiae = GetMinutiaeList();
            Cylinder[] cylinders;
            int length;
            createTemplate(minutiae, minutiae.Length, out cylinders, out length);
        }

        private Minutia[] GetMinutiaeList()
        {
            List<Minutia> minutiaeList = new List<Minutia>();

            for (int i = 1; i <= 100; i++)
            {
                Random random = new Random();
                Minutia minutia;
                minutia.X = i;
                minutia.Y = i;
                minutia.Angle = random.Next(-3, 3) + (float)random.NextDouble();
                minutiaeList.Add(minutia);
            }
            return minutiaeList.ToArray();
        }
    }
}
