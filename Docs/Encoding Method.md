# Encoding Method

## Error Correction
### Error Correction Goal
8bit의 payload에 대해 최대 8bit의 비트 단위 오류 정정

### Codeword Size
48 bits 

## Encoding Method 1
Error Correction을 수행하여 생성된 48bits의 codeword를 이용해 2개의 RGB값을 생성  
C0 = C[0:8]  
C1 = C[8:16]  
C2 = C[16:24]  
C3 = C[24:32]  
C4 = C[32:40]  
C5 = C[40:48]  

1st RGB(R, G, B): (C0, C1, C2)  
2nd RGB(R, G, B): (C3, C4, C5)  

## Decoding
주어진 RGB 값 2개를 사용해 48bits의 codeword로 변환 후, 원본 Payload 복원  
1st RGB(R, G, B): (C0, C1, C2)  
2nd RGB(R, G, B): (C3, C4, C5)  

C0 = C[0:8]  
C1 = C[8:16]  
C2 = C[16:24]  
C3 = C[24:32]  
C4 = C[32:40]  
C5 = C[40:48]  

## Image Generation
Encoding 과정을 통해 계산한 2개의 RGB값을 사용하여 $2\times 2$ 의  image를 생성  
1st 2nd  
2nd 1st


## Encoding Method 2
\boxed{
\text{CubeID}
=
\left\lfloor \frac{B}{64} \right\rfloor \cdot 64
+
\left\lfloor \frac{G}{32} \right\rfloor \cdot 8
+
\left\lfloor \frac{R}{32} \right\rfloor
}

## Decoding 
주어진 RGB 값을 기반으로 Encoding 과정의 역변환 진행