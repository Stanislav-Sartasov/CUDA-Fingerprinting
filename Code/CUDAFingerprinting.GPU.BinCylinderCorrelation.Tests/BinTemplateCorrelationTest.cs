using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common.BinCylinderCorrelation.Tests;
using CUDAFingerprinting.Common.BinCylinderCorrelation;
using System.Runtime.InteropServices;

namespace CUDAFingerprinting.GPU.BinCylinderCorrelation.Tests
{
    [StructLayout(LayoutKind.Sequential)]
    public struct CylinderC
    {
        public IntPtr values;
        public uint valuesCount;
        public Single angle;
        public Single norm;
        public uint templateIndex;
    }

    [TestClass]
    public class BinTemplateCorrelationTest
    {
        [DllImport("CUDAFingerprinting.GPU.BinCylinderCorrelation.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "initMCC")]
        public static extern void initMCC(
	        IntPtr cylinderDb, uint cylinderDbCount,
	        IntPtr templateDbLengths, uint templateDbCount);

        [DllImport("CUDAFingerprinting.GPU.BinCylinderCorrelation.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "processMCC")]
        public static extern IntPtr processMCC(
            IntPtr query, uint queryLength,
            uint cylinderDbCount, uint templateDbCount);

        [DllImport("CUDAFingerprinting.GPU.BinCylinderCorrelation.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "checkValsFromTest")]
        public static extern uint checkValsFromTest(
            IntPtr query, uint queryLength,
            uint cylinderDbCount, uint templateDbCount);

        public int cylindersPerTemplate = 8;

        [TestMethod]
        public void TestBinTemplateCorrelationMassive()
        {
            string homeFolder = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            BinTemplateCorrelationTests.ParseDb(homeFolder + "\\mcc_cs_db.txt");
            BinTemplateCorrelationTests.ParseQuery(homeFolder + "\\mcc_cs_query.txt");

            CylinderDatabase cylinderDb = new CylinderDatabase(
                BinTemplateCorrelationTests.contiguousCylinders, 
                BinTemplateCorrelationTests.templateIndices);

            int cylinderSize = Marshal.SizeOf(typeof(CylinderC));
            
            IntPtr db = Marshal.AllocHGlobal(BinTemplateCorrelationTests.contiguousCylinders.Length * cylinderSize);
            IntPtr curDbPtr = new IntPtr(db.ToInt32());
            for (int i = 0; i < BinTemplateCorrelationTests.contiguousCylinders.Length; i++)
            {
                CylinderC curDbCylinder = new CylinderC();
                curDbCylinder.values = Marshal.AllocHGlobal(cylindersPerTemplate * sizeof(int));
                int[] curManagedValues = new int[cylindersPerTemplate];
                for (int j = 0; j < cylindersPerTemplate; j++)
                {
                    curManagedValues[j] = unchecked((int)BinTemplateCorrelationTests.contiguousCylinders[i].Values[j]);
                }
                Marshal.Copy(curManagedValues, 0, curDbCylinder.values, cylindersPerTemplate);

                curDbCylinder.angle = (float)BinTemplateCorrelationTests.contiguousCylinders[i].Angle;
                curDbCylinder.norm = (float)BinTemplateCorrelationTests.contiguousCylinders[i].Norm;
                curDbCylinder.valuesCount = (uint)BinTemplateCorrelationTests.contiguousCylinders[i].Values.Length;
                curDbCylinder.templateIndex = BinTemplateCorrelationTests.templateIndices[i];

                Marshal.StructureToPtr(curDbCylinder, curDbPtr, false);
                curDbPtr = new IntPtr(curDbPtr.ToInt32() + Marshal.SizeOf(typeof(CylinderC)));
            }

            IntPtr query = Marshal.AllocHGlobal(BinTemplateCorrelationTests.query.Cylinders.Length * cylinderSize);
            IntPtr curQueryPtr = new IntPtr(query.ToInt32());
            for (int i = 0; i < BinTemplateCorrelationTests.query.Cylinders.Length; i++)
            {
                CylinderC curQueryCylinder = new CylinderC();
                curQueryCylinder.values = Marshal.AllocHGlobal(cylindersPerTemplate * sizeof(int));
                int[] curManagedValues = new int[cylindersPerTemplate];
                for (int j = 0; j < cylindersPerTemplate; j++)
                {
                    curManagedValues[j] = unchecked((int)BinTemplateCorrelationTests.query.Cylinders[i].Values[j]);
                }
                Marshal.Copy(curManagedValues, 0, curQueryCylinder.values, cylindersPerTemplate);

                curQueryCylinder.angle = (float)BinTemplateCorrelationTests.query.Cylinders[i].Angle;
                curQueryCylinder.norm = (float)BinTemplateCorrelationTests.query.Cylinders[i].Norm;
                curQueryCylinder.valuesCount = (uint)BinTemplateCorrelationTests.query.Cylinders[i].Values.Length;
                curQueryCylinder.templateIndex = 0; // Always 0 for query, just for the sake of completeness, doesn't really matter

                Marshal.StructureToPtr(curQueryCylinder, curQueryPtr, false);
                curQueryPtr = new IntPtr(curQueryPtr.ToInt32() + Marshal.SizeOf(typeof(CylinderC)));
            }

            IntPtr templateDbLengths = Marshal.AllocHGlobal(BinTemplateCorrelationTests.templateDbLengths.Length * sizeof(int));
            Marshal.Copy(BinTemplateCorrelationTests.templateDbLengths, 0, templateDbLengths, BinTemplateCorrelationTests.templateDbLengths.Length);

            initMCC(
                db, (uint)BinTemplateCorrelationTests.contiguousCylinders.Length,
                templateDbLengths, (uint)BinTemplateCorrelationTests.templateDbLengths.Length);

            IntPtr similaritiesPtr = processMCC(
                query, (uint)BinTemplateCorrelationTests.query.Cylinders.Length,
                (uint)BinTemplateCorrelationTests.contiguousCylinders.Length, (uint)BinTemplateCorrelationTests.templateDbLengths.Length);

            float[] similarities = new float[BinTemplateCorrelationTests.templateDbLengths.Length];

            Marshal.Copy(similaritiesPtr, similarities, 0, similarities.Length);

            for (int i = 0; i < similarities.Length; i++)
            {
                Console.Write(similarities[i] + (i != similarities.Length - 1 ? ", " : ""));
            }
            Console.WriteLine();
        }
    }
}
