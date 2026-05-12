#include "Adafruit_SHT4x.h"
#include "max6675.h"

Adafruit_SHT4x sht4 = Adafruit_SHT4x();

float Tsum=0.;
float Tksum=0.;
float Hsum=0.;
float n=0.;



int thermoDO = 4;
int thermoCS = 5;
int thermoCLK = 6;

MAX6675 thermocouple(thermoCLK, thermoCS, thermoDO);



void setup() {
  Serial.begin(9600);
  while (!Serial)
      delay(10);     // will pause Zero, Leonardo, etc until serial console opens
  
  Serial.println("MAX6675 and SHT4 test");
  // wait for MAX chip to stabilize
  delay(500);

 
  while(!sht4.begin() ) {
      Serial.println("Couldn't find SHT4x");
      delay(1000);
  }

  Serial.println("Found SHT4x sensor");
  Serial.print("Serial number 0x");
  Serial.println(sht4.readSerial(), HEX);

  // You can have 3 different precisions, higher precision takes longer
  sht4.setPrecision(SHT4X_HIGH_PRECISION);
  switch (sht4.getPrecision()) {
     case SHT4X_HIGH_PRECISION: 
       Serial.println("High precision");
       break;
     case SHT4X_MED_PRECISION: 
       Serial.println("Med precision");
       break;
     case SHT4X_LOW_PRECISION: 
       Serial.println("Low precision");
       break;
  }

  // You can have 6 different heater settings
  // higher heat and longer times uses more power
  // and reads will take longer too!
  sht4.setHeater(SHT4X_NO_HEATER);
  switch (sht4.getHeater()) {
     case SHT4X_NO_HEATER: 
       Serial.println("No heater");
       break;
     case SHT4X_HIGH_HEATER_1S: 
       Serial.println("High heat for 1 second");
       break;
     case SHT4X_HIGH_HEATER_100MS: 
       Serial.println("High heat for 0.1 second");
       break;
     case SHT4X_MED_HEATER_1S: 
       Serial.println("Medium heat for 1 second");
       break;
     case SHT4X_MED_HEATER_100MS: 
       Serial.println("Medium heat for 0.1 second");
       break;
     case SHT4X_LOW_HEATER_1S: 
       Serial.println("Low heat for 1 second");
       break;
     case SHT4X_LOW_HEATER_100MS: 
       Serial.println("Low heat for 0.1 second");
       break;
  }
  

}


void loop() {
  // basic readout test, just print the current temp
  
//  Serial.print("C = "); 
 // Serial.println(thermocouple.readCelsius());
  
  //read SHT$ sensor
  sensors_event_t humidity, temp;
  
  //uint32_t timestamp = millis();
  sht4.getEvent(&humidity, &temp);// populate temp and humidity objects with fresh data
  //timestamp = millis() - timestamp;

  Tsum=Tsum+temp.temperature;
  Tksum=Tksum+thermocouple.readCelsius();
  Hsum=Hsum+humidity.relative_humidity;
  n=n+1;
   
  if (n==5){
    
    Serial.print("Temp: "); Serial.print(Tsum/n); Serial.print("  Humid: "); Serial.print(Hsum/n); Serial.print(" k_temp: "); Serial.println(Tksum/n);
    n=0.;
    Hsum=0.;
    Tsum=0.;
    Tksum=0.;
  }

  // For the MAX6675 to update, you must delay AT LEAST 250ms between reads!
  delay(2000);
}


