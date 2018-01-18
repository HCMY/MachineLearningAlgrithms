#include<iostream>

const double breaker = 0.0001;

const int NewtonMethod(unsigned long long int number){
    double Xn;
    double Xnn = 10.00;
    do
    {
        Xn = Xnn;
        Xnn= Xn-0.5*((Xn*Xn-number)/Xn);
        std::cout<<Xnn<<std::endl;
    }while(std::abs(Xn-Xnn)>breaker);

    return (int)Xnn;
}


int main(){
    unsigned long long int number = 10000000000;
    //std::cin>>number;
    //std::cout<<std::sqrt(number)<<std::endl;
    std::cout<<NewtonMethod(number)<<std::endl;
}