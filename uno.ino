#include "Wire.h"
#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"

MPU6050 mpu;

bool dmpReady = false;
uint8_t fifoBuffer[64];

Quaternion q;
VectorFloat gravity;
VectorInt16 aa;
VectorInt16 aaReal;
VectorInt16 aaWorld;

const float ACC_SCALE = 9.80665 / 16384.0;
const float GYRO_SCALE = 1.0 / 131.0;

unsigned long lastTime = 0;
const unsigned long dtMicros = 10000;  // 100 Hz


void setup() {
  Serial.begin(115200);
  Wire.begin();

  mpu.initialize();

  if (!mpu.testConnection()) {
    Serial.println("ERROR:MPU_NOT_FOUND");

    while (1) {
      delay(1000);
    }
  }

  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);

  uint8_t devStatus = mpu.dmpInitialize();

  // Your existing calibration
  mpu.setXGyroOffset(220);
  mpu.setYGyroOffset(76);
  mpu.setZGyroOffset(-85);
  mpu.setZAccelOffset(1788);

  if (devStatus == 0) {
    mpu.setDMPEnabled(true);
    dmpReady = true;
  } else {
    Serial.print("ERROR:DMP_");
    Serial.println(devStatus);
  }
}


void loop() {

  if (!dmpReady) {
    return;
  }

  unsigned long now = micros();

  if (now - lastTime < dtMicros) {
    return;
  }

  lastTime = now;

  if (mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) {

    // Quaternion
    mpu.dmpGetQuaternion(&q, fifoBuffer);

    // Raw acceleration
    int16_t ax, ay, az;

    mpu.getAcceleration(&ax, &ay, &az);

    aa.x = ax;
    aa.y = ay;
    aa.z = az;

    // Gravity
    mpu.dmpGetGravity(&gravity, &q);

    // Linear acceleration in body frame
    mpu.dmpGetLinearAccel(&aaReal, &aa, &gravity);

    // World-frame acceleration
    mpu.dmpGetLinearAccelInWorld(&aaWorld, &aaReal, &q);

    // Gyroscope
    VectorInt16 gyroDMP;

    mpu.dmpGetGyro(&gyroDMP, fifoBuffer);

    Serial.print("GYRx=");
    Serial.print(gyroDMP.x * GYRO_SCALE, 3);

    Serial.print(" GYRy=");
    Serial.print(gyroDMP.y * GYRO_SCALE, 3);

    Serial.print(" GYRz=");
    Serial.print(gyroDMP.z * GYRO_SCALE, 3);

    Serial.print(" ACCx_body=");
    Serial.print(aaReal.x * ACC_SCALE, 3);

    Serial.print(" ACCy_body=");
    Serial.print(aaReal.y * ACC_SCALE, 3);

    Serial.print(" ACCz_body=");
    Serial.print(aaReal.z * ACC_SCALE, 3);

    Serial.println();
  }
}