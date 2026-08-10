#include <Wire.h>
#include <Adafruit_MLX90640.h>
#include <math.h>

Adafruit_MLX90640 mlx;

#define PIR_PIN 13

float frame[32 * 24];

const float TEMP_OFFSET = 3.5;       
const int HOT_PIXEL_THRESHOLD = 12;

// Rolling baseline ambient (captured only when PIR is LOW)
float baselineAmbient = NAN;
const float BASELINE_ALPHA = 0.05;    // Slow-moving average (higher = faster adapt)

void setup()
{
  Serial.begin(115200);
  delay(1000);

  pinMode(PIR_PIN, INPUT);

  Wire.begin(21, 22);

  if (!mlx.begin(MLX90640_I2CADDR_DEFAULT, &Wire))
  {
    Serial.println("MLX90640 not detected!");
    while (1);
  }

  mlx.setMode(MLX90640_CHESS);
  mlx.setRefreshRate(MLX90640_4_HZ);

  Serial.println();
  Serial.println("===============================");
  Serial.println("MLX90640 + PIR Ready");
  Serial.println("===============================");
}

void loop()
{
  int pir = digitalRead(PIR_PIN);

  if (pir == LOW)
  {
    // No motion — safe to update baseline ambient
    if (mlx.getFrame(frame) == 0)
    {
      float currentAvg = calculateAmbient(frame);

      if (isnan(baselineAmbient))
      {
        baselineAmbient = currentAvg;   // First reading, seed it directly
      }
      else
      {
        baselineAmbient = (BASELINE_ALPHA * currentAvg) +
                           ((1.0 - BASELINE_ALPHA) * baselineAmbient);
      }
    }

    Serial.print("PIR: No Motion   Baseline Ambient: ");
    Serial.print(isnan(baselineAmbient) ? 0.0 : baselineAmbient, 2);
    Serial.println(" °C");
    Serial.println("Detection: 0");
    delay(100);
    return;
  }

  // Motion detected — read thermal frame and compare against baseline
  if (mlx.getFrame(frame) != 0)
  {
    Serial.println("Thermal Read Error");
    delay(100);
    return;
  }

  // If we never got a baseline yet, fall back to current frame average
  float ambient = isnan(baselineAmbient) ? calculateAmbient(frame) : baselineAmbient;

  float maxTemp = findMax(frame);
  int hotPixels = countHotPixels(frame, ambient);
  int badPixels = countBadPixels(frame);

  Serial.print("Ambient(baseline) : ");
  Serial.print(ambient, 2);

  Serial.print(" °C   Max : ");
  Serial.print(maxTemp, 2);

  Serial.print(" °C   Hot Pixels : ");
  Serial.print(hotPixels);

  Serial.print("   Bad Pixels : ");
  Serial.println(badPixels);

  if ((maxTemp > ambient + TEMP_OFFSET) &&
      (hotPixels >= HOT_PIXEL_THRESHOLD))
  {
    Serial.println("Human Detected");
    Serial.println("Detection: 1");
  }
  else
  {
    Serial.println("No Human");
    Serial.println("Detection: 0");
  }

  Serial.println("------------------------------");

  delay(250);
}

float calculateAmbient(float *frame)
{
  float sum = 0.0;
  int valid = 0;

  for (int i = 0; i < 768; i++)
  {
    if (!isnan(frame[i]))
    {
      sum += frame[i];
      valid++;
    }
  }

  if (valid == 0)
    return 0;

  return sum / valid;
}

float findMax(float *frame)
{
  float maximum = -1000;

  for (int i = 0; i < 768; i++)
  {
    if (!isnan(frame[i]) && frame[i] > maximum)
    {
      maximum = frame[i];
    }
  }

  return maximum;
}

int countHotPixels(float *frame, float ambient)
{
  int count = 0;
  float threshold = ambient + TEMP_OFFSET;

  for (int i = 0; i < 768; i++)
  {
    if (!isnan(frame[i]) && frame[i] > threshold)
    {
      count++;
    }
  }

  return count;
}

int countBadPixels(float *frame)
{
  int bad = 0;

  for (int i = 0; i < 768; i++)
  {
    if (isnan(frame[i]))
      bad++;
  }

  return bad;
}
