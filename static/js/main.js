document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const submitButton = document.getElementById('submitButton');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');

    // Default values for categorical fields
    const categoricalDefaults = {
        MSSubClass: "60",
        MSZoning: "RL",
        Street: "Pave",
        Alley: "NA",
        LotShape: "Reg",
        LandContour: "Lvl",
        Utilities: "AllPub",
        LotConfig: "Inside",
        LandSlope: "Gtl",
        Neighborhood: "NAmes",
        Condition1: "Norm",
        Condition2: "Norm",
        BldgType: "1Fam",
        HouseStyle: "2Story",
        RoofStyle: "Gable",
        RoofMatl: "CompShg",
        Exterior1st: "VinylSd",
        Exterior2nd: "VinylSd",
        MasVnrType: "None",
        ExterQual: "TA",
        ExterCond: "TA",
        Foundation: "PConc",
        BsmtQual: "TA",
        BsmtCond: "TA",
        BsmtExposure: "No",
        BsmtFinType1: "Unf",
        BsmtFinType2: "Unf",
        Heating: "GasA",
        HeatingQC: "TA",
        CentralAir: "Y",
        Electrical: "SBrkr",
        KitchenQual: "TA",
        Functional: "Typ",
        FireplaceQu: "NA",
        GarageType: "Attchd",
        GarageFinish: "Unf",
        GarageQual: "TA",
        GarageCond: "TA",
        PavedDrive: "Y",
        PoolQC: "NA",
        Fence: "NA",
        MiscFeature: "NA",
        SaleType: "WD",
        SaleCondition: "Normal"
    };

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading spinner
        loadingSpinner.classList.add('active');
        submitButton.disabled = true;
        resultDiv.style.display = 'none';
        errorDiv.classList.remove('active');
        
        // Get form data
        const formData = new FormData(form);
        const data = {
            // Required fields with defaults
            Id: 1,
            FirstFlrSF: 1000.0,
            SecondFlrSF: 500.0,
            ThreeSsnPorch: 0.0,
            MSZoning: "RL",
            
            // Add all categorical defaults
            ...categoricalDefaults
        };
        
        // Add form values, converting numbers where appropriate
        for (let [key, value] of formData.entries()) {
            if (value !== '') {
                // Convert numeric strings to numbers
                if (!isNaN(value) && value !== '') {
                    value = Number(value);
                }
                data[key] = value;
            }
        }

        // Ensure all required numeric fields have defaults
        const numericDefaults = {
            GrLivArea: 1500.0,
            TotalBsmtSF: 1000.0,
            LotArea: 8000,
            OverallQual: 5,
            OverallCond: 5,
            YearBuilt: 1970,
            FullBath: 2,
            HalfBath: 1,
            BedroomAbvGr: 3,
            TotRmsAbvGrd: 6,
            GarageCars: 2,
            GarageArea: 400,
            GarageYrBlt: 1970,
            YearRemodAdd: 1970,
            Fireplaces: 0,
            PoolArea: 0,
            BsmtFullBath: 0,
            BsmtHalfBath: 0
        };

        // Add numeric defaults for empty fields
        for (const [key, defaultValue] of Object.entries(numericDefaults)) {
            if (!(key in data) || data[key] === '') {
                data[key] = defaultValue;
            }
        }
        
        try {
            console.log('Sending data:', data);  // Debug log
            
            // Make API call
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                console.error('API Error:', errorData);  // Add detailed error logging
                throw new Error(errorData.detail || 'API request failed');
            }
            
            const result = await response.json();
            
            // Display result
            resultDiv.innerHTML = `
                <div class="alert alert-success">
                    <h4>Predicted House Price:</h4>
                    <p class="display-4">${result.predicted_price_formatted}</p>
                    ${result.note ? `<p class="text-muted">${result.note}</p>` : ''}
                </div>
            `;
            resultDiv.style.display = 'block';
            
        } catch (error) {
            console.error('Error:', error);
            errorDiv.textContent = error.message || 'An error occurred while making the prediction. Please try again.';
            errorDiv.classList.add('active');
        } finally {
            // Hide loading spinner
            loadingSpinner.classList.remove('active');
            submitButton.disabled = false;
        }
    });
}); 