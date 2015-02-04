// @(#)root/tmva $Id$
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodNN                                                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      NeuralNetwork                                                             *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Peter Speckmayer      <peter.speckmayer@gmx.at> - CERN, Switzerland       *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodNN
#define ROOT_TMVA_MethodNN

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodNN                                                             //
//                                                                      //
// Neural Network implementation                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TTree
#include "TTree.h"
#endif
#ifndef ROOT_TRandom3
#include "TRandom3.h"
#endif
#ifndef ROOT_TH1F
#include "TH1F.h"
#endif
#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif


namespace TMVA {

    class MethodNN : public MethodBase
   {

   public:

      // standard constructors
      MethodNN ( const TString& jobName,
                 const TString&  methodTitle,
                 DataSetInfo& theData,
                 const TString& theOption,
                 TDirectory* theTargetDir = 0 );

      MethodNN ( DataSetInfo& theData,
                 const TString& theWeightFile,
                 TDirectory* theTargetDir = 0 );

      virtual ~MethodNN();

      virtual Bool_t HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets );

      void Train();

      Double_t GetMvaValue( Double_t* err=0, Double_t* errUpper=0 );

      using MethodBase::ReadWeightsFromStream;

      // write weights to stream
      void AddWeightsXMLTo     ( void* parent ) const;

      // read weights from stream
      void ReadWeightsFromStream( std::istream & i );
      void ReadWeightsFromXML   ( void* wghtnode );

      // ranking of input variables
      const Ranking* CreateRanking();

      // nice output
      void PrintCoefficients( void );

   protected:

      // make ROOT-independent C++ class for classifier response (classifier-specific implementation)
      void MakeClassSpecific( std::ostream&, const TString& ) const;

      // get help message text
      void GetHelpMessage() const;


   private:

      // the option handling methods
      void DeclareOptions();
      void ProcessOptions();

      // general helper functions
      void     Init();


   private:
      TString  fLayout;
      TString  fErrorStrategy;
      TString  fTrainingStrategy;
      



      ClassDef(MethodNN,0) // neural network 
   };

} // namespace TMVA

#endif
