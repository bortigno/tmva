#pragma once

#include "TApplication.h"
#include "TCanvas.h"
#include "TSystem.h"
//#include <stdio.h>

namespace TMVA
{
    class  Monitoring
    {

    public:
        Monitoring (int argc, char* /*argv[]*/)
            : fArgc (argc)
        {
            fArgv = new char[argc] ();
        }    

        Monitoring ()
        {
        }    

        ~Monitoring () 
        { 
            delete fCanvas; 
            delete[] fArgv;
        }

        void Start ()
        {
            fApplication = new TApplication ("TMVA Monitoring", &fArgc, &fArgv);
            fApplication->SetReturnFromRun (true);

            fCanvas = new TCanvas ("TMVA Monitoring", "Monitoring", 1000, 500);
        }


        void ProcessEvents ()
        {
            GetCanvas ()->Modified();
            GetCanvas ()->Update();
            gSystem->ProcessEvents(); //canvas can be edited during the loop
        }

        TCanvas* GetCanvas () { return fCanvas; }

    private:
        TCanvas* fCanvas;
        TApplication* fApplication;

        int fArgc;
        char* fArgv;
    };
}
