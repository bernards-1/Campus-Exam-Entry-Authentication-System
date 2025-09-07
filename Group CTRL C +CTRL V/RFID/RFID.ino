#include <SPI.h>
#include <MFRC522.h>

#define SS_PIN 5   // SDA
#define RST_PIN 22 // RST

MFRC522 mfrc522(SS_PIN, RST_PIN); // Create MFRC522 instance

void setup() {
  Serial.begin(115200);
  SPI.begin();            // Init SPI bus
  mfrc522.PCD_Init();     // Init MFRC522
  Serial.println("Scan a card...");
}

void loop() {
  // 如果没有新卡
  if (!mfrc522.PICC_IsNewCardPresent()) return;

  // 如果没有读到卡号
  if (!mfrc522.PICC_ReadCardSerial()) return;

  // 打印 UID
  String uidStr = "";
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    if (mfrc522.uid.uidByte[i] < 0x10) Serial.print("0");
    uidStr += String(mfrc522.uid.uidByte[i], HEX);
  }
  uidStr.toUpperCase();

  Serial.println(uidStr);

  // 停止通信
  mfrc522.PICC_HaltA();
  mfrc522.PCD_StopCrypto1();
}
