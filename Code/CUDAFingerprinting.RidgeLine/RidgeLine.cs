using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Design;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using CUDAFingerprinting.Common;
using Microsoft.Win32;

namespace CUDAFingerprinting.RidgeLine
{
    enum Directions { Forward, Back}
    enum MinutiaTypes { NotMinutia, LineEnding, Intersection}
    class Minutia
    {
        public int X;
        public int Y;
        public double Angle;
        public MinutiaTypes Type;

        public Minutia()
        {
            
        }
        public Minutia(int x, int y, double angle, MinutiaTypes type)
        {
            X = x;
            Y = y;
            Angle = angle;
            Type = type;
        }
    }
    internal class RidgeLine
    {
        private int[,] _image;
        private PixelwiseOrientationField _orientation;
        private int _step;
        private int _wings;
        private bool _diffAngle;
        public bool[,] _visited;
        private const int BuildUp = 1000; //specially for trade coordinates between methods
        private const double _pi4 = Math.PI/4;

        public int[] _section;
        private double _angleSection;

        public List<Minutia> Minutias;

        public RidgeLine(int[,] image, int step, int wings)
        {
            _image = image;
            _orientation = new PixelwiseOrientationField(image, 16);
            _step = step;
            _wings = wings;
            _section = new int[2*wings + 1];
            _diffAngle = false;
            _visited = new bool[image.GetLength(0), image.GetLength(1)];
            Minutias = new List<Minutia>();
        }

        private void NewSection(int mid)
        {
            int i = mid/BuildUp;
            int j = mid%BuildUp;


            double angle = _orientation.GetOrientation(i, j) + Math.PI/2;

            _section[_wings] = mid;

            for (var k = 1; k <= _wings; k++)
            {
                int xs, ys, xe, ye;
                xs = Convert.ToInt32(i - k*Math.Cos(angle));
                ys = Convert.ToInt32(j - k*Math.Sin(angle));
                xe = Convert.ToInt32(i + k*Math.Cos(angle));
                ye = Convert.ToInt32(j + k*Math.Sin(angle));

                if (xs < 0 || ys < 0 || xs >= _image.GetLength(1) || ys >= _image.GetLength(0))
                {
                    //ys = -1;
                    //xs = 0;
                    _section[_wings] = -1;
                    break;
                }

                if (xe < 0 || ye < 0 || xe >= _image.GetLength(1) || ye >= _image.GetLength(0))
                {
                    //ye = -1;
                    //xe = 0;
                    _section[_wings] = -1;
                    break;
                }

                _section[_wings - k] = xs*BuildUp + ys;
                _section[_wings + k] = xe*BuildUp + ye;
            }
        } 

        private int[] FindEdges()
        {
            int lPoint = _wings;
            int rPoint = _wings;

            bool check = false;

            while (!check)
            {
                int x = _section[lPoint]/BuildUp;
                int y = _section[lPoint]%BuildUp;

                //if (x < 0 || y < 0 || x >= _image.GetLength(1) || y >= _image.GetLength(0))
                //{
                //    if (lPoint < _wings) lPoint++;
                //    break;
                //}
                if (_section[lPoint] == -1)
                {
                    lPoint++;
                    break;
                }
                if (_image[y, x] > 125)
                {
                    lPoint++;
                    check = true;
                }
                else
                {
                    lPoint--;

                    if (_section[lPoint] == -1)
                    {
                        lPoint++;
                        check = true;
                    }

                    if (lPoint == 0)
                    {
                        if (_image[_section[lPoint]%BuildUp, _section[lPoint]/BuildUp] > 125)
                            lPoint++;
                        break;
                }
            }
            }

            while (check)
            {
                int x = _section[rPoint]/BuildUp;
                int y = _section[rPoint]%BuildUp;

                //if (x < 0 || y < 0 || x >= _image.GetLength(1) || y >= _image.GetLength(0))
                //{
                //    if (rPoint > _wings) rPoint--;
                //    break;
                //}

                if (_section[rPoint] == -1)
                {
                    rPoint--;
                    break;
                }

                if (_image[y, x] > 125)
                {
                    rPoint--;
                    check = false;
                }
                else
                {
                    rPoint++;

                    if (_section[rPoint] == -1)
                    {
                        //rPoint++;
                        rPoint--;
                        check = false;
                    }

                    if (rPoint == _wings*2)
                    {
                        if (_image[_section[rPoint]%BuildUp, _section[rPoint]/BuildUp] > 125)
                            rPoint--;
                        break;
                    }
                }
            }

            var res = new int[3] {_section[lPoint], _section[rPoint], _section[(rPoint + lPoint) / 2]};

            return res;
        }

        private int MakeStep(int startPoint, Directions direction)
        {
            int x = startPoint/BuildUp;
            int y = startPoint%BuildUp;


            double angle = _orientation.GetOrientation(x, y) + ((int)direction) * Math.PI + (_diffAngle ? Math.PI : 0) + Math.PI * 2;
            while (angle > Math.PI*2) angle -=Math.PI*2;
            double angCheck = _orientation.GetOrientation(x, y);
            x += (int) (_step*Math.Cos(angle) + 0.5);
            y += (int) (_step*Math.Sin(angle) + 0.5);

            if (x < 0 || y < 0 || x >= _image.GetLength(1) || y >= _image.GetLength(0))
            {
                return -1;
            }

            double angle2 = _orientation.GetOrientation(x, y) + ((int)direction) * Math.PI + (_diffAngle ? Math.PI : 0) + Math.PI * 2;
            double ang2check = _orientation.GetOrientation(x, y);
            while (angle2 > Math.PI * 2) angle2 -= Math.PI * 2;
            //if (Math.Abs(angle - angle2) > _pi4)
            if (Math.Abs(angCheck - ang2check) > Math.PI / 2.0)
            {
                _diffAngle = !_diffAngle;
            }


            return x*BuildUp + y;
        }

        private void Paint(int[] edges1, int[] edges2)
        {
            List<int> lX = new List<int>() { edges1[0] / BuildUp, edges1[1] / BuildUp, edges2[0] / BuildUp, edges2[1] / BuildUp};
            List<int> lY = new List<int>() { edges1[0] % BuildUp, edges1[1] % BuildUp, edges2[0] % BuildUp, edges2[1] % BuildUp};

            int xMax = lX.Max();
            int xMin = lX.Min();
            int yMax = lY.Max();
            int yMin = lY.Min();

            for (int i = xMin; i <= xMax; i++)
            {
                for (int j = yMin; j <= yMax; j++)
                {
                    if(_image[j,i] < 30) _visited[j, i] = true;
                }
            }
        }

        private MinutiaTypes CheckStopCriteria(int threshold = 175)
        {
            int xcenter = _section[_wings] / BuildUp;
            int ycenter = _section[_wings] % BuildUp;
            if (_visited[ycenter, xcenter])
            {
                MakeTestBmp(xcenter, ycenter);
                return MinutiaTypes.Intersection;
            }
                

            

            double mean = 0;
            for (int i = 0; i < _wings*2 + 1; i++)
            {
                int xcur = _section[i]/BuildUp;
                int ycur = _section[i] % BuildUp;
                mean += _image[ycur, xcur];
            }
            mean /= (_wings*2 + 1);
            if (mean > threshold)
            {
                MakeTestBmp(xcenter, ycenter);
                return MinutiaTypes.LineEnding;
            }

            return MinutiaTypes.NotMinutia;
        }

        void MakeTestBmp(int x, int y)
        {
            int[,] image = new int[_visited.GetLength(0), _visited.GetLength(1)];

            for (int i = 0; i < image.GetLength(1); i++)
            {
                for (int j = 0; j < image.GetLength(0); j++)
                {
                    if (_visited[j, i])
                    {
                        image[j, i] = 255;
                    }
                }
            }

            var bmp = ImageHelper.SaveArrayToBitmap(image);

            bmp.SetPixel(x, 364 - y, Color.Red);

            bmp.Save("Test1.bmp");
        }

        Minutia FollowLine(int startPoint, Directions direction)
        {
            NewSection(startPoint);
            if (_section[_wings] == -1) return null;
            MinutiaTypes minutiaType;
            int xcur, ycur;
            double angle;
            do
            {
                xcur = startPoint / BuildUp;
                ycur = startPoint % BuildUp;

                angle = _orientation.GetOrientation(xcur, ycur);
                angle += ((int) direction)*Math.PI + (_diffAngle ? Math.PI : 0);
                while (angle > Math.PI*2) angle -=Math.PI*2;
                var edges = FindEdges();

                startPoint = MakeStep(edges[2], direction);
                if (startPoint < 0) return null;
                NewSection(startPoint);
                if (_section[_wings] == -1) return null;
                minutiaType = CheckStopCriteria();
                Paint(edges, FindEdges());
            } while (minutiaType == MinutiaTypes.NotMinutia);
            xcur = startPoint / BuildUp;
            ycur = startPoint % BuildUp;
            Minutia res = new Minutia(xcur, ycur, angle, minutiaType);
            return res;
        }

        public List<Minutia> GetMinutiaList()
        {
            return Minutias;
        }

        public void FindMinutiaLine(int startPoint, double duplicateDelta, int colorThreshold = 30)
        {
            int x = startPoint/BuildUp;
            int y = startPoint%BuildUp;

            if (_image[y, x] < colorThreshold && !_visited[y, x])
            {
                _visited[y, x] = true;

                var minutia1 = FollowLine(startPoint, Directions.Forward);
                if (minutia1 != null)
                {
                if (minutia1.Type == MinutiaTypes.LineEnding)//WRONG: there would be no minutia if we arrive to the vicinity of this point even amount of times.
                    {
                    if (IsDuplicate(minutia1, duplicateDelta))
                        CheckAndDeleteFalseMinutia(minutia1, duplicateDelta);
                    else
                        Minutias.Add(minutia1);
                    }
                else
                {
                    if (!IsDuplicate(minutia1, duplicateDelta))
                        Minutias.Add(minutia1);
                }
                }
                _diffAngle = false;

                var minutia2 = FollowLine(startPoint, Directions.Back);
                if (minutia2 != null)
                {
                if (minutia2.Type == MinutiaTypes.LineEnding)//WRONG: there would be no minutia if we arrive to the vicinity of this point even amount of times.
                {
                    if (IsDuplicate(minutia2, duplicateDelta))
                        CheckAndDeleteFalseMinutia(minutia2, duplicateDelta);
                    else
                        Minutias.Add(minutia2);
                }
                else
                {
                    if (!IsDuplicate(minutia2, duplicateDelta))
                        Minutias.Add(minutia2);
                }
        }
            }

            _diffAngle = false;
        }

        void CheckAndDeleteFalseMinutia(Minutia minutia, double delta)
        {
            var i =
                Minutias.FindIndex(x => Math.Sqrt(Math.Pow(x.X - minutia.X, 2) + Math.Pow(x.Y - minutia.Y, 2)) < delta);

            Minutias.RemoveAt(i);
        }

        bool IsDuplicate(Minutia minutia, double delta)
        {
            return Minutias.Exists(x => Math.Sqrt(Math.Pow(x.X - minutia.X, 2) + Math.Pow(x.Y - minutia.Y, 2)) < delta && minutia.Type == x.Type);
        }
    }

}