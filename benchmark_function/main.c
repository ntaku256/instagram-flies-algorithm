#include<stdio.h> 
#include<stdlib.h> 
#include<time.h> 
#define ALPHA 0.03 

int kn=0,mn=0,an=0; 
float r_GU=1.0, r_CYOKI=1.0, r_PA=1.0; 

float frand(void){ 
    return (float)rand()/RAND_MAX;
} 

int hand(){ 
    float gu, cyoki, pa; 

    gu=r_GU*frand(); 
    cyoki=r_CYOKI*frand(); 
    pa=r_PA*frand(); 

 
    if(gu>cyoki){ 
        if(gu>pa) return 0; 
        else return 2; 
    } 
    else{ 
        if(cyoki>pa)return 1; 
        else return 2; 
    } 
} 

 

void winlose(int AI_t, int G_t){ 
    if(AI_t==0){ 
        if(G_t==1){ 
            r_GU=r_GU+ALPHA*r_GU; 
            kn++; 
        } 
        else if(G_t==2){ 
            r_GU=r_GU-ALPHA*r_GU; 
            mn++; 
        } 
        else 
            an++; 
    } 
    else if(AI_t==1){ 
        if(G_t==2){ 
            r_CYOKI=r_CYOKI+ALPHA*r_CYOKI; 
            kn++; 
        } 
        else if (G_t==0){ 
            r_CYOKI=r_CYOKI-ALPHA*r_CYOKI; 
            mn++; 
        } 
        else 
            an++; 
    } 
    else if(AI_t==2){
	    if(G_t==0){
		    r_PA=r_PA+ALPHA*r_PA;
		    kn++;
	   }
	   else if (G_t==1){
		    r_PA=r_PA-ALPHA*r_PA;
		    mn++;
	   }
	   else
	        an++;
	}
} 

  

int main(void){
    int i,n=100;
    int AI_t,G_t;
                
    srand((unsigned)time(NULL));
        
    printf("対戦を開始します\n\n");
    while(n!=9){
        
        printf("終了したい場合、９を入力しなさい\n");

        AI_t=hand( );

        printf("ゴリラの手を入力しなさい\n");
	    scanf("%d",&G_t);	
	    if(G_t==9){
	        n=9;
        }
	    if(G_t<0 || G_t>2){
	        printf("エラーです\n");
	    }

        winlose(AI_t,G_t);

        printf("AIの手＝%d  ゴリラの手＝%d\n",AI_t, G_t);
	    printf("r_GUの割合＝%6.3f, r_CYOKIの割合=%6.3f, r_PAの割合=%6.3f\n\n",  r_GU,r_CYOKI,r_PA);
    }	
    
    printf("AIは勝った回数は%d回です\n",kn);
	printf("AIは負けた回数は%d回です\n",mn);
	printf("あいこの回数は%d回です\n\n",an);
	return 0;
}
