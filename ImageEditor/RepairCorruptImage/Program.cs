using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace RepairCorruptImage
{
    class Program
    {
        const string RESULTS_PATH = @"C:\Users\ag4488\Documents\Visual Studio 2019\Projects\Machine Learning\Image Processing\ImageEditor\GenerateCorruptImage\Results";
        static void Main(string[] args)
        {
            var workingImg = new Bitmap($"{RESULTS_PATH}\\wine_dataset_img_corrupt.png", true);
            int x, y;
            int countOutliers=0;            
            
            // Loop through the images pixels to get the columns median (average) values
            // for making up for outlying pixels (eg red)
            var colSimpleAvgArray = new double[12];
            var colSimpleCountArray = new int[12];
            for (x = 0; x < workingImg.Height; x++)
                for (y = 0; y < workingImg.Width; y++)
                {
                    Color pixelColor = workingImg.GetPixel(y,x);
                    var redt = pixelColor.R;
                    var greent = pixelColor.G;
                    var bluet = pixelColor.B;
                    if ((redt == greent) && (greent == bluet)){
                        colSimpleAvgArray[y] = colSimpleAvgArray[y] + redt;
                        colSimpleCountArray[y] = colSimpleCountArray[y] + 1;
                    }
                }  
            
            // Replace the outliers (eg red pixels) with average 
            for (x = 0; x < workingImg.Height; x++)
            {
                for (y = 0; y < workingImg.Width; y++)
                {
                    Color pixelColor = workingImg.GetPixel(y, x);
                    var redt = pixelColor.R;
                    var greent = pixelColor.G;
                    var bluet = pixelColor.B;

                    if ((redt == 255) && (greent == 0) && (bluet == 0))
                    {
                        countOutliers++;
                        // replace outlier with column simple average
                        var colSimpleAverage = (int)Math.Round(colSimpleAvgArray[y]/colSimpleCountArray[y]);
                        workingImg.SetPixel(y,x,Color.FromArgb(colSimpleAverage, colSimpleAverage, colSimpleAverage));
                    }
                }
            }
            Console.WriteLine($"#Outliers fixed: {countOutliers}");

            workingImg.Save($"{RESULTS_PATH}\\wine_dataset_img_repaired.png");
            Console.WriteLine($"{RESULTS_PATH}\\wine_dataset_img_repaired.png");
        }
    }
}
