using System;
using System.Collections.Generic;
using System.ComponentModel;
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
    class Minutia
    {
        public int X;
        public int Y;
        private double Angle;
        public int Type;
        public Minutia(int x, int y, double angle,int type)
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
        private OrientationField _orientation;
        private int _step;
        private int _wings;
        private bool _diffAngle;
        private bool[,] _visited;
        private const int BuildUp = 1000; //specially for trade coordinates between methods
        private const double _pi4 = Math.PI/4;

        public int[] _section;

        public List<Minutia> Minutias ;

        public RidgeLine(int[,] image, int step, int wings)
        {
            _image = image;
            _orientation = new OrientationField(image);
            _step = step;
            _wings = wings;
            _section = new int[2*wings + 1];
            _diffAngle = false;
            _visited = new bool[image.GetLength(0), image.GetLength(1)];
        }

        private void NewSection(int mid)
        {
            int i = mid/BuildUp;
            int j = mid%BuildUp;

            double angle = _orientation.GetOrientation(i, j) + Math.PI/2;

            _section[_wings] = i*BuildUp + j;

            for (var k = 1; k <= _wings; k++)
            {
                int xs, ys, xe, ye;
                xs = Convert.ToInt32(i - k*Math.Cos(angle));
                ys = Convert.ToInt32(j - k*Math.Sin(angle));
                xe = Convert.ToInt32(i + k*Math.Cos(angle));
                ye = Convert.ToInt32(j + k*Math.Sin(angle));

                _section[_wings - k] = xs*BuildUp + ys;
                _section[_wings + k] = xe*BuildUp + ye;
            }
        } //All right

        private int[] FindEdges()
        {
            int lPoint = _wings;
            int rPoint = _wings;

            bool check = false;

            while (!check)
            {
                int x = _section[lPoint]/BuildUp;
                int y = _section[lPoint]%BuildUp;

                if (_image[x, y] > 125)
                {
                    check = true;
                }
                else
                {
                    lPoint--;

                    if (lPoint < 0) break;
                }
            }

            while (check)
            {
                int x = _section[rPoint]/BuildUp;
                int y = _section[rPoint]%BuildUp;

                if (_image[x, y] > 125)
                {
                    check = false;
                }
                else
                {
                    rPoint--;

                    if (rPoint > _wings*2) break;
                }
            }

            var res = new int[2] {lPoint, rPoint};

            return res;
        }

        private int MakeStep(int startPoint, Directions direction)//direction should be 0 or 1
        {
            int x = startPoint/BuildUp;
            int y = startPoint%BuildUp;


            double angle = _orientation.GetOrientation(x, y) + (_diffAngle ? Math.PI : 0) + ((int)direction) * Math.PI;

            x += (int) (_step*Math.Cos(angle) + 0.5);
            y += (int) (_step*Math.Sin(angle) + 0.5);

            double angle2 = _orientation.GetOrientation(x, y);

            if (Math.Abs(angle - angle2) > _pi4)
            {
                _diffAngle = true;
            }
            else
            {
                _diffAngle = false;
            }

            return x*BuildUp + y;
        }

        private void Paint(int lstartPoint, int rstartPoint, double angle, int stepDistX, int stepDistY)
        {
            int xd = (int) (Math.Cos(angle) + 0.5);
            int yd = (int) (Math.Sin(angle) + 0.5);

            for (int i = lstartPoint; i <= rstartPoint; i++)
            {
                int xcur = _section[i]/BuildUp;
                int ycur = _section[i]%BuildUp;

                int xLimit = xcur + stepDistX;
                int yLimit = ycur + stepDistY;

                while ((ycur < yLimit) && (xcur < xLimit))
                {
                    _visited[ycur, xcur] = true;
                    ycur += yd;
                    xcur += xd;
                }
            }
        }

        private int CheckStopCriteria(int threshold = 200)
        {
            double mean = 0;
            for (int i = 0; i < _wings*2 + 1; i++)
                mean += _section[i];
            mean /= (_wings*2 + 1);
            if (mean > threshold) return 1; //line ends

            int xcenter = _section[_wings]/BuildUp;
            int ycenter = _section[_wings]%BuildUp;
            if (_visited[ycenter, xcenter]) return 2; //intersection

            return 0; //no minutia
        }

        public Minutia FollowLine(int startPoint, Directions direction)
        {
            NewSection(startPoint);
            int flag;
            int xcur, ycur;
            double angle;
            do
            {
                xcur = startPoint / BuildUp;
                ycur = startPoint % BuildUp;
                angle = _orientation.GetOrientation(xcur, ycur) + (_diffAngle ? Math.PI : 0) + ((int)direction) * Math.PI;

                var edges = FindEdges();
                Paint(edges[0], edges[1], angle, (int) (_step*Math.Cos(angle) + 0.5), (int) (_step*Math.Sin(angle) + 0.5));

                startPoint = MakeStep(startPoint, direction);
                NewSection(startPoint);
                flag = CheckStopCriteria();
            } while (flag == 0);
            Minutia res = new Minutia(xcur, ycur, angle, flag);
            return res;
        }

        public Minutia[] FindMinutiaLine(int startPoint, double duplicateDelta, int threshold = 30)
        {
            int x = startPoint/BuildUp;
            int y = startPoint%BuildUp;
            if (_image[y, x] < threshold)
            {
                var minutia1 = FollowLine(startPoint, Directions.Forward);
                if (minutia1.Type == 1)
                {
                    if (!(IsDuplicate(minutia1, duplicateDelta)))
                    {
                        Minutias.Add(minutia1);
                    }
                }
                var minutia2 = FollowLine(startPoint, Directions.Back);
                return (new Minutia[2] {minutia1, minutia2});
            }
        }

        public bool IsDuplicate(Minutia minutia, double delta)
        {
            return Minutias.Exists(x => Math.Sqrt(Math.Pow(x.X - minutia.X, 2) + Math.Pow(x.Y - minutia.Y, 2)) < delta);
        }
    }

}