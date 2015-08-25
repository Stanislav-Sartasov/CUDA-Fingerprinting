using System;
using System.IO;

namespace CUDAFingerprinting.GPU.LocalSimilaritySort.Test
{
    public class LocalScoresTable<T>
    {
        public T[,] data;
        public short[] templateSizes;

        public short QueryTemplateSize
        {
            get
            {
                return (short)data.GetLength(1);
            }
        }
        public int TemplatesNumber
        {
            get
            {
                return templateSizes.GetLength(0);
            }
        }

        // File format:
        // database width (short queryTemplateSize) \n
        // number of cylinders in each template (short[] templateSizes) \n
        // local similarities databse (T[,] data)
        public void ReadFile(string fileName)
        {
            var input = new StreamReader(fileName);

            int queryTemplateSize = short.Parse(input.ReadLine());

            string[] auxilaryStrings = input.ReadLine().Split(' ');
            int templatesNumber = auxilaryStrings.GetLength(0);

            templateSizes = new short[templatesNumber];

            for (int i = 0; i < templatesNumber; ++i)
                templateSizes[i] = short.Parse(auxilaryStrings[i]);

            int databaseHeight = 0;
            foreach (int size in templateSizes)
                databaseHeight += size;

            data = new T[databaseHeight, queryTemplateSize];

            for (int i = 0; i < databaseHeight; ++i)
            {
                auxilaryStrings = input.ReadLine().Split(' ');

                for (int j = 0; j < queryTemplateSize; ++j)
                    if (typeof(T) == typeof(float))
                        data[i, j] = (T)Convert.ChangeType(float.Parse(auxilaryStrings[j]), typeof(T));
                    else /*if(typeof(T) == typeof(short)*/
                        data[i, j] = (T)Convert.ChangeType(short.Parse(auxilaryStrings[j]), typeof(T));
            }
        }

        public void RandGenerate()
        {
            Random generator = new Random();

            int queryTemplateSize = generator.Next(60, 100);

            int templatesNumber = generator.Next(200);

            templateSizes = new short[templatesNumber];
            int databaseHeight = 0;
            for (int i = 0; i < templatesNumber; ++i)
            {
                templateSizes[i] = (short)generator.Next(60, 100);
                databaseHeight += templateSizes[i];
            }

            data = new T[databaseHeight, queryTemplateSize];
            for (int i = 0; i < databaseHeight; ++i)
                for (int j = 0; j < queryTemplateSize; ++j)
                    if (typeof(T) == typeof(float))
                        data[i, j] = (T)Convert.ChangeType(generator.NextDouble(), typeof(T));
                    else /*if(typeof(T) == typeof(short))*/
                        data[i, j] = (T)Convert.ChangeType(generator.Next(0, 64), typeof(T));
        }

        public void SaveDBToFile(string fileName)
        {
            var output = new StreamWriter(fileName);

            var rrr = data.GetLength(0).ToString() + " " + QueryTemplateSize.ToString();
            output.WriteLine(rrr);
            
            uint dbHeight = 0;
            foreach(short size in templateSizes)
                dbHeight += (uint)size;

            for (uint i = 0; i < dbHeight; ++i)
            {
                string line = string.Empty;
                for (short j = 0; j < QueryTemplateSize; ++j)
                    line += data[i, j].ToString() + " ";

                output.WriteLine(line);
            }
            output.Close();
        }

        public void SaveTemplateSizesToFile(string fileName)
        {
            var output = new StreamWriter(fileName);

            output.WriteLine(TemplatesNumber.ToString());

            string line = string.Empty;

            for (uint i = 0; i < TemplatesNumber; ++i)
                line += templateSizes[i].ToString() + " ";

            output.Write(line);
            output.Close();
        }
    }
}
