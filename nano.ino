void setup() {
  Serial.begin(115200);
}

void loop() {
  int flex1 = analogRead(A0);
  int flex2 = analogRead(A1);
  int flex3 = analogRead(A2);
  int flex4 = analogRead(A3);
  int flex5 = analogRead(A6);

  Serial.print("F1=");
  Serial.print(flex1);

  Serial.print(" F2=");
  Serial.print(flex2);

  Serial.print(" F3=");
  Serial.print(flex3);

  Serial.print(" F4=");
  Serial.print(flex4);

  Serial.print(" F5=");
  Serial.println(flex5);

  delay(10);//~100 Hz
}